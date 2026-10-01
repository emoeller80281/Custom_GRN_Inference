from pathlib import Path
import h5py, numpy as np, pandas as pd

ROOT = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PROC = ROOT / "data/processed"
GT = ROOT / "data/ground_truth_files/cell_type_specific"
OUT = ROOT / "TETHER/new_testing_results/celltype_sizes.csv"

TISSUE = {
    ("mESC", None): "gastrulation embryo",
    ("kidney", None): "kidney",
    ("mouse_liver", None): "liver",
    ("GSE246464_HSC", None): "bone marrow",
    ("10x_E18_mouse_brain", None): "embryonic brain",
    ("GSE140203_shareseq", "brain"): "brain",
    ("GSE140203_shareseq", "skin_late_anagen"): "skin",
}

def read_obs_col(h5, col):
    g = h5[f"mod/rna/obs/{col}"]
    if isinstance(g, h5py.Group):  # categorical
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]], dtype=object)
        codes = g["codes"][:]
        return pd.Series(np.where(codes >= 0, cats[np.clip(codes, 0, None)], None))
    return pd.Series([v.decode() if isinstance(v, bytes) else v for v in g[:]])

rows, skipped = [], []
for dataset_dir in sorted(p for p in PROC.iterdir() if p.is_dir()):
    dataset = dataset_dir.name
    gt_dir = GT / "mm10" / dataset
    for sample_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        h5mu = sample_dir / "multiome_processed.h5mu"
        if not h5mu.exists():
            skipped.append(f"{dataset}:{sample_dir.name} (no multiome_processed.h5mu)")
            continue
        if not gt_dir.exists():
            skipped.append(f"{dataset}:{sample_dir.name} (no GT dir)")
            continue
        with h5py.File(h5mu, "r") as h5:
            counts = read_obs_col(h5, "celltype").value_counts()
        label_map = GT / f"{dataset}_label_map.tsv"
        if label_map.exists():
            lm = pd.read_csv(label_map, sep="\t", comment="#")
            members = lm.groupby("celltype_group")["celltype"].apply(list).to_dict()
        else:
            members = {}
        tissue = TISSUE.get((dataset, sample_dir.name), TISSUE.get((dataset, None), dataset))
        for gt_path in sorted(gt_dir.glob("*_ground_truth.parquet")):
            slice_name = gt_path.name.removesuffix("_ground_truth.parquet")
            labels = members.get(slice_name, [slice_name, slice_name.replace("_", " ")])
            n_cells = int(counts.reindex(labels).fillna(0).sum())
            gt = pd.read_parquet(gt_path, columns=["Source"])
            rows.append(dict(dataset_name=f"{dataset}:{sample_dir.name}", tissue_type=tissue,
                             cell_type=slice_name, num_cells=n_cells,
                             num_gt_tfs=gt["Source"].str.upper().nunique()))
        unmatched = set(counts.index) - {l for s in rows if s["dataset_name"] == f"{dataset}:{sample_dir.name}"
                                          for l in members.get(s["cell_type"], [s["cell_type"], s["cell_type"].replace("_", " ")])}
        if unmatched:
            print(f"{dataset}:{sample_dir.name} annotated types without GT: {sorted(unmatched)}")

df = pd.DataFrame(rows)
never = df.groupby(df["dataset_name"].str.split(":").str[0] + "/" + df["cell_type"])["num_cells"].sum()
print("GT files with no cells in any sample:", sorted(never[never == 0].index))
# Shared GT dirs (GSE140203) list cell types from other tissues; keep only slices with cells.
df = df[df["num_cells"] > 0]
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(len(df), "rows")
print("Skipped:", *skipped, sep="\n  ")
