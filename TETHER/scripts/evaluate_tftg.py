"""Fixed-edge diagnostics for SimpleTFTGRegulationModel + TFTGEdgeBagDataset.

Requires torch, numpy, pandas, matplotlib, seaborn. Uses loader.dataset directly
(no loader workers/sampler). Pass indices to evaluate your original training edges.
The model must accept batch['binding_score'] as in the simplified cached model.
No training or weight changes. Returned figures/results can be used in notebooks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import default_collate


def evaluate_tftg(model, train_loader, indices=None, n_per_class=4,
                  repeats=100, quantiles=41, seed=123, out_dir=None, show=True):
    """Run baseline, cell resampling, independent shuffles, means and quantiles.

    indices: positional indices into train_loader.dataset (recommended).
    Defaults to the first n_per_class examples per label having candidate peaks.
    Resampling draws from the SAME pool, not a held-out cell population.
    out_dir: optionally save all result CSVs and plot PNGs.
    Returns {'tables': ..., 'figures': ..., 'indices': ...}.
    """
    ds = train_loader.dataset
    if indices is None:
        e = ds.inputs
        has_peaks = e['peak_atac_cols'].map(len).gt(0).to_numpy()
        indices = [int(i) for label in (1, 0) for i in
                   np.flatnonzero(e['label'].eq(label).to_numpy() & has_peaks)[:n_per_class]]
    indices = list(indices)
    if not indices or repeats < 1 or quantiles < 2:
        raise ValueError('Select edges, repeats >= 1 and quantiles >= 2.')
    device = next(model.parameters()).device
    old_modes = [(m, m.training) for m in model.modules()]
    old_resample, old_rng = ds.resample_cells, ds._rng
    ds.resample_cells, ds._rng = True, np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    tables, figures = {}, {}

    def draw():
        b = default_collate([ds[i] for i in indices])
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}

    def predict(b):
        z, cells = model(b)
        return z.detach().cpu().numpy(), cells.detach().cpu().numpy()

    def record(b, condition, repeat=0):
        z, _ = predict(b)
        return pd.DataFrame({
            'Item': range(B), 'Edge': names, 'Label': labels,
            'Condition': condition, 'Repeat': repeat,
            'Logit': z, 'Probability': 1 / (1 + np.exp(-np.clip(z, -700, 700))),
            'Baseline logit': base, 'Logit change': z - base,
            'Margin drop': (2 * labels - 1) * (base - z),
            'Correct': (z >= 0) == labels.astype(bool),
        })

    try:
        model.eval()
        with torch.no_grad():
            batch = draw()
            B, C, P = batch['peak_accessibility'].shape
            labels = batch['label'].cpu().numpy()
            names = [f"{i}: {batch['tf_name'][i]} → {batch['tg_id'][i]} "
                     f"({'+' if labels[i] else '−'})" for i in range(B)]
            cells = [torch.where(batch['cell_mask'][i].bool())[0] for i in range(B)]
            peaks = [torch.where(batch['peak_mask'][i].bool())[0] for i in range(B)]
            # Frozen binding is unchanged by every intervention; compute once.
            binding = torch.zeros(B, P, device=device)
            for i, ps in enumerate(peaks):
                if len(ps):
                    binding[i, ps] = model.tf_peak_model(
                        tf_embedding=batch['tf_embedding'][i:i+1].expand(len(ps), -1, -1),
                        tf_mask=batch['tf_mask'][i:i+1].expand(len(ps), -1),
                        peak_embedding=batch['peak_sequences'][i, ps].float(),
                    ).reshape(-1).sigmoid()
            batch['binding_score'] = binding
            base, base_cells = predict(batch)
            tables['baseline'] = record(batch, 'Original')
            tables['baseline']['Dataset index'] = indices
            tables['baseline']['n_cells'] = [len(c) for c in cells]
            tables['baseline']['n_peaks'] = [len(p) for p in peaks]
            tables['baseline']['Sample ID'] = batch['sample_id']
            tables['baseline']['Cell type'] = batch['cell_type']
            tables['cell_logits'] = pd.DataFrame([
                {'Item': i, 'Edge': names[i], 'Cell index': int(cell_id), 'Logit': float(z)}
                for i in range(B) for cell_id, z in zip(
                    batch['cell_indices'][i, cells[i]].cpu().tolist(),
                    base_cells[i, cells[i].cpu().numpy()])])

            original_sets = [set(batch['cell_indices'][i, cells[i]].cpu().tolist())
                             for i in range(B)]
            rows = []
            for r in range(repeats):
                b = draw()
                b['binding_score'] = binding
                result = record(b, 'Resampled cells', r)
                bags = [tuple(sorted(b['cell_indices'][i][b['cell_mask'][i].bool()]
                                     .cpu().tolist())) for i in range(B)]
                result['Cell set'] = bags
                result['Training cell overlap'] = [
                    len(set(bag) & original_sets[i]) / len(bag) for i, bag in enumerate(bags)]
                rows.append(result)
            tables['resampling'] = pd.concat(rows, ignore_index=True)
            # Overlap is with the fresh baseline bag, not necessarily the training bag.
            tables['resampling'].rename(columns={
                'Training cell overlap': 'Baseline cell overlap'}, inplace=True)
            tables['resampling_summary'] = tables['resampling'].groupby('Edge', sort=False).agg(
                unique_cell_sets=('Cell set', 'nunique'), accuracy=('Correct', 'mean'),
                median_probability=('Probability', 'median'),
                p05=('Probability', lambda x: x.quantile(.05)),
                p95=('Probability', lambda x: x.quantile(.95)),
                mean_absolute_logit_change=('Logit change', lambda x: x.abs().mean()),
                mean_baseline_cell_overlap=('Baseline cell overlap', 'mean')).reset_index()

            features = {'TF expression': 'tf_expression', 'TG expression': 'tg_expression',
                        'Accessibility': 'peak_accessibility'}
            groups = {name: [key] for name, key in features.items()}
            groups['All independently'] = list(features.values())
            for mode, count in [('shuffle', repeats), ('mean', 1)]:
                rows = []
                for name, keys in groups.items():
                    for r in range(count):
                        b = dict(batch)
                        for key in keys:
                            b[key] = batch[key].clone()
                            for i, cs in enumerate(cells):
                                values = batch[key][i, cs]
                                if mode == 'shuffle':
                                    order = torch.randperm(len(cs), device=device, generator=generator)
                                    b[key][i, cs] = values[order]
                                else:
                                    b[key][i, cs] = values.mean(0, keepdim=True).expand_as(values)
                        condition = 'All three' if mode == 'mean' and len(keys) == 3 else name
                        rows.append(record(b, condition, r))
                tables[mode] = pd.concat(rows, ignore_index=True)

            qs = torch.linspace(0, 1, quantiles, device=device)
            values = {i: torch.quantile(batch['peak_accessibility'][i][cs][:, peaks[i]]
                                        .float(), qs, dim=0)
                      for i, cs in enumerate(cells) if len(peaks[i])}
            rows = []
            for j, q in enumerate(qs.cpu().tolist()):
                b = dict(batch)
                b['peak_accessibility'] = batch['peak_accessibility'].clone()
                for i, v in values.items():
                    b['peak_accessibility'][i][cells[i][:, None], peaks[i][None, :]] = v[j]
                result = record(b, 'Accessibility quantile')
                result = result[result['Item'].isin(values)].copy()
                result['Quantile'] = q
                result['Assigned accessibility'] = [float(values[i][j].mean()) for i in result.Item]
                result['Per-peak values'] = [values[i][j].cpu().tolist() for i in result.Item]
                rows.append(result)
            tables['sweep'] = pd.concat(rows, ignore_index=True)
    finally:
        ds.resample_cells, ds._rng = old_resample, old_rng
        for module, mode in old_modes:
            module.training = mode

    # Plot helpers: all comparisons retain the same edge order.
    def panels(title, count):
        fig, ax = plt.subplots(count, 1, figsize=(max(10, 1.4 * B), 3 * count),
                               squeeze=False, sharex=True)
        figures[title] = fig
        return fig, ax.ravel()

    def distributions(table, column, title):
        conditions = table.Condition.unique()
        fig, axes = panels(title, len(conditions))
        for ax, condition in zip(axes, conditions):
            sns.boxplot(data=table[table.Condition.eq(condition)], x='Edge', y=column,
                        order=names, color='lightsteelblue', showfliers=True, ax=ax)
            ax.axhline(.5 if column == 'Probability' else 0, color='gray', ls=':')
            if column == 'Probability':
                ax.scatter(range(B), tables['baseline'].Probability, c='black', marker='D', s=20)
                ax.set_ylim(-.03, 1.03)
            ax.set(title=condition, xlabel='')
        axes[-1].tick_params(axis='x', rotation=45)
        fig.tight_layout()

    def heatmap(table, title):
        matrix = table.pivot_table(index='Condition', columns='Edge', values='Margin drop',
                                    aggfunc='mean', sort=False).reindex(columns=names)
        limit = max(abs(matrix.to_numpy()).max(), 1e-6)
        fig, ax = plt.subplots(figsize=(max(10, 1.4 * B), 4))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-limit, vmax=limit, cbar_kws={'label': 'Correct-label margin drop'}, ax=ax)
        ax.set(title=f'{title}: red weakens the correct prediction', xlabel='', ylabel='')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        figures[title] = fig

    fig, axes = panels('baseline', 2)
    axes[0].scatter(range(B), tables['baseline'].Probability,
                    c=['tab:blue' if y else 'tab:red' for y in labels])
    axes[0].axhline(.5, c='gray', ls=':')
    axes[0].set(ylabel='Probability', ylim=(-.03, 1.03), title='Baseline predictions')
    sns.boxplot(data=tables['cell_logits'], x='Edge', y='Logit', order=names, ax=axes[1])
    axes[1].axhline(0, c='gray', ls=':')
    axes[1].set(xlabel='', title='Baseline real-cell logits')
    axes[1].tick_params(axis='x', rotation=45)
    fig.tight_layout()
    for test in ('resampling', 'shuffle'):
        for column in ('Probability', 'Logit change'):
            distributions(tables[test], column, f'{test}_{column.replace(" ", "_")}')
    for test in ('shuffle', 'mean'):
        heatmap(tables[test], f'{test}_margin')
    fig, axes = panels('mean_probabilities', 4)
    for ax, (condition, g) in zip(axes, tables['mean'].groupby('Condition', sort=False)):
        ax.scatter(range(B), tables['baseline'].Probability, c='black', marker='D', label='Original')
        ax.scatter(range(B), g.Probability, c=['tab:blue' if y else 'tab:red' for y in labels],
                    label='Mean replaced')
        ax.axhline(.5, c='gray', ls=':')
        ax.set(title=condition, ylabel='Probability', ylim=(-.03, 1.03))
    axes[0].legend()
    axes[-1].set_xticks(range(B), names, rotation=45, ha='right')
    fig.tight_layout()
    for column in ('Quantile', 'Assigned accessibility'):
        fig, axes = plt.subplots((B + 3)//4, min(4, B), figsize=(16, 3.3*((B+3)//4)),
                                  squeeze=False, sharey=True)
        for i, ax in enumerate(axes.ravel()):
            if i >= B:
                ax.set_visible(False)
                continue
            g = tables['sweep'].loc[lambda t: t.Item.eq(i)]
            ax.plot(g[column], g.Logit, '-o', ms=3, c='tab:blue' if labels[i] else 'tab:red')
            ax.axhline(base[i], c='black', ls='--', label='Original')
            ax.axhline(0, c='gray', ls=':')
            ax.set(title=names[i], xlabel=column, ylabel='Edge logit')
            if g.empty:
                ax.text(.5, .5, 'No candidate peaks', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        figures[f'sweep_{column.replace(" ", "_")}'] = fig
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(path / f'{name}.csv', index=False)
        for name, fig in figures.items():
            fig.savefig(path / f'{name}.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return {'tables': tables, 'figures': figures, 'indices': indices}
