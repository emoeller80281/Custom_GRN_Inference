import contextlib
import os
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

"""
This script uses the ProstT5 model from Rostlab to create combined amino acid (AA)
sequence and 3Di structure embeddings for a set of transcription factors (TFs).

This is done by loading the AA sequence for each TF from a FASTA file, matching it to a
corresponding 3Di sequence from a Foldseek output FASTA file, and then using the ProstT5
tokenizer and model to create a combined embedding. The resulting embeddings are saved as
PyTorch tensors in the specified output directory.

This script is from the TFBindFormer project. See the original code and documentation here:
https://github.com/BioinfoMachineLearning/TFBindFormer

Read the original paper here:
https://www.biorxiv.org/content/10.64898/2026.04.09.717563v2


Why this runs in three stages
-----------------------------
ProstT5 emits 1024 dims for the AA pass and 1024 for the 3Di pass, concatenated to 2048
per residue. That is too wide to hold as a padded [n_tfs, max_len, d] table on the GPU, so
it has to be reduced -- but the reduction has to be the SAME function for every TF, or the
embeddings are not comparable to each other.

An earlier version built

    proj = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Linear(1024, d_model))

*inside* the per-TF function, with no seed. Every TF was therefore projected through a
freshly initialised random network, i.e. into its own random basis. Each TF's embedding
stayed internally coherent (first-half vs second-half mean cosine 0.80) while every
cross-TF relationship was destroyed. Measured on the resulting cache:

    SD of cosine between mean-pooled TFs          0.0883   (1/sqrt(128) = 0.0884)
    mean |cos| between per-TF top PC directions   0.0707   (independent bases predict 0.071)
    AUROC separating same-DBD-family TF pairs      0.497   (chance)

So: run ProstT5 once and keep the raw 2048-d output (`--stage raw`), fit ONE PCA across
the residues of every TF of every species (`--stage fit`), then apply that single saved
projection to all of them (`--stage project`). The projection is written to disk so a TF
embedded months later lands in the same basis instead of a new one.

Run `scripts/validate_tf_embeddings.py` afterwards -- it re-measures the three numbers
above and checks that orthologs and DBD families come out where they should.
"""

PROSTT5_DIM = 2048  # 1024 (AA pass) + 1024 (3Di pass), concatenated per residue
# The FASTA basenames already end in "_protein" (Adnp_protein.fasta), so these are
# appended to that: Adnp_protein_raw.pt -> Adnp_protein_embedding.pt. That output name
# is what build_tf_to_dna_train_data.py globs for ("*_protein_embedding.pt"), and it is
# what the previous version of this script wrote -- so a rerun overwrites the old file
# in place. Appending "_protein_*" here instead produces Adnp_protein_protein_*.pt,
# which still matches that glob but does NOT collide with the old name: the directory
# then holds two generations at once and every consumer silently reads both.
RAW_SUFFIX = "_raw.pt"
OUT_SUFFIX = "_embedding.pt"


# ---------------------------------------------------------------------------
# Stage 1: ProstT5 -> raw 2048-d per-residue embeddings
# ---------------------------------------------------------------------------

def _encode_one(tokenizer, model, prefixed_seq, device):
    """Encode a single prefixed sequence, returning its [1 + L + 1, 1024] hidden states.

    The AA and 3Di passes are run separately rather than as a batch of two. T5 encoder
    self-attention never crosses the batch dimension and padded positions are masked, so
    this is numerically identical to the batched call -- but attention is quadratic in
    length, and halving the batch halves peak activation memory. That is what lets the
    longest proteins (5,588 residues for mm10) fit on a V100 instead of needing an A100.
    """
    ids = tokenizer(
        [prefixed_seq],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    # CPU autocast to float16 is not a supported fast path; the CPU fallback below runs
    # the model in float32 instead.
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode():
        with autocast:
            outputs = model(ids.input_ids, attention_mask=ids.attention_mask)

    return outputs.last_hidden_state[0]


def embedding_features(tokenizer, model, seq_1d, seq_3di, device):
    """Raw ProstT5 features for one protein: [L, 2048], L = min(len(AA), len(3Di)).

    No projection here on purpose -- see the module docstring. Anything that reduces the
    width has to be shared across every TF, which cannot be decided from inside a
    single-protein function.
    """
    d1 = len(seq_1d)
    d2 = len(seq_3di)

    # preprocess sequences
    seq_1d = " ".join(list(re.sub(r"[UZOB]", "X", seq_1d)))
    seq_3di = " ".join(list(seq_3di.lower()))

    # Add special tokens to indicate sequence type for the model
    hidden_aa = _encode_one(tokenizer, model, "<AA2fold> " + seq_1d, device)
    hidden_3di = _encode_one(tokenizer, model, "<fold2AA> " + seq_3di, device)

    # Extract the embeddings for the AA and 3Di sequences (last hidden state), skipping
    # the leading special token
    emb_aa = hidden_aa[1 : d1 + 1]
    emb_3di = hidden_3di[1 : d2 + 1]

    # Trim the two sequences to the same length and concatenate along the feature dimension
    L = min(d1, d2)
    emb = torch.cat(
        [emb_aa[:L], emb_3di[:L]], dim=-1
    ).float()

    return emb.cpu()


def extract_np_accession(text):
    match = re.search(r"(NP_\d+\.\d+)", text)
    return match.group(1) if match else None


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run_raw_stage(args):
    """ProstT5 over every FASTA in --aa_dir, saving [L, 2048] fp16 tensors to --raw_dir.

    Attention cost is quadratic in protein length and the length distribution has a long
    tail -- median 484 residues but a maximum of 5,588, with only ~30 of 1,523 proteins
    over 2,000. On a smaller GPU those few can exhaust VRAM. Rather than lose the whole
    run to them (the batch script uses `set -e`), each OOM falls back to CPU for that one
    protein: slow, but it runs a handful of times at most and keeps the output complete.
    """
    from transformers import T5Tokenizer, T5EncoderModel

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    # Built lazily on the first OOM -- a second full copy of the model costs ~12 GB of
    # host RAM, which is not worth paying on runs that never need it.
    cpu_model = None
    n_cpu_fallback = 0

    aa_dir = Path(args.aa_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load the ProstT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5",
        revision="refs/pr/2",
        do_lower_case=False,
    )

    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5",
        revision="refs/pr/2",
        use_safetensors=True,
    ).to(device)

    if device.type == "cpu":
        model.float()
    else:
        model.half()

    # Load 3Di sequences into a dictionary keyed by accession number (e.g. NP_123456.1)
    logging.info(f"Loading 3Di sequences from {args.di_fasta}")
    di_dict = {}
    for rec in SeqIO.parse(args.di_fasta, "fasta"):
        header = f"{rec.id} {rec.description}"
        acc = extract_np_accession(header) or rec.id.split()[0]
        di_dict[acc] = str(rec.seq)

    n_done = n_skipped = n_missing = 0

    for fname in sorted(os.listdir(aa_dir)):
        if not fname.endswith(".fasta"):
            continue

        # The TF files should be named like "TFNAME.fasta"
        tf_id = fname.replace(".fasta", "")
        out_path = raw_dir / f"{tf_id}{RAW_SUFFIX}"

        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        # Load the AA sequence for this TF
        aa_rec = next(SeqIO.parse(aa_dir / fname, "fasta"))

        # Extract the NP accession number for the TF so it can be matched to the 3Di sequence
        aa_header = f"{aa_rec.id} {aa_rec.description}"
        acc = extract_np_accession(aa_header) or tf_id

        if acc not in di_dict:
            logging.warning(f"No 3Di for {tf_id} (acc: {acc}), skipping")
            n_missing += 1
            continue

        aa_seq = str(aa_rec.seq)

        try:
            emb = embedding_features(tokenizer, model, aa_seq, di_dict[acc], device)
        except Exception as exc:
            if not _is_oom(exc):
                raise

            logging.warning(
                f"{tf_id} ({len(aa_seq)} residues) did not fit in GPU memory; "
                f"falling back to CPU for this protein."
            )
            torch.cuda.empty_cache()

            if cpu_model is None:
                logging.info("  loading a CPU copy of ProstT5 for the fallback path...")
                cpu_model = T5EncoderModel.from_pretrained(
                    "Rostlab/ProstT5",
                    revision="refs/pr/2",
                    use_safetensors=True,
                ).float().eval()

            emb = embedding_features(
                tokenizer, cpu_model, aa_seq, di_dict[acc], torch.device("cpu")
            )
            n_cpu_fallback += 1

        # fp16 halves the intermediate footprint (~2.6 GB mm10, ~5.2 GB hg38 at fp32) and
        # costs nothing: ProstT5 already ran in half precision on GPU.
        torch.save(emb.half(), out_path)
        n_done += 1

        if n_done % 50 == 0:
            logging.info(f"  {n_done} embedded ({tf_id}: {tuple(emb.shape)})")

    logging.info(
        f"Raw stage done: {n_done} embedded, {n_skipped} already present, "
        f"{n_missing} without a 3Di match. -> {raw_dir}"
    )
    if n_cpu_fallback:
        logging.info(
            f"  {n_cpu_fallback} protein(s) were embedded on CPU after a GPU OOM. "
            f"Their output is identical, just slower to produce."
        )


# ---------------------------------------------------------------------------
# Stage 2: fit ONE projection across every TF of every species
# ---------------------------------------------------------------------------

def _raw_files(raw_dirs):
    files = []
    for d in raw_dirs:
        found = sorted(Path(d).glob(f"*{RAW_SUFFIX}"))
        if not found:
            raise FileNotFoundError(f"No {RAW_SUFFIX} files in {d}. Run --stage raw first.")
        files.extend(found)
    return files


def run_fit_stage(args):
    """Fit one PCA over residues pooled from every raw directory, and save it."""
    from sklearn.decomposition import PCA

    files = _raw_files(args.raw_dir)
    logging.info(f"Fitting projection over {len(files)} TFs from {len(args.raw_dir)} directories")

    rng = np.random.default_rng(args.seed)

    # Sample residues rather than loading every one: the fit only needs to see the
    # covariance structure, and 200k x 2048 fp32 is already 1.6 GB.
    per_file = max(1, args.fit_max_residues // len(files))
    sampled = []

    for path in files:
        emb = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]
        if emb.shape[0] > per_file:
            rows = rng.choice(emb.shape[0], size=per_file, replace=False)
            emb = emb[rows]
        sampled.append(emb)

    X = np.concatenate(sampled, axis=0)
    del sampled
    logging.info(f"  fitting PCA({args.d_model}) on {X.shape[0]:,} residues x {X.shape[1]} dims")

    pca = PCA(
        n_components=args.d_model,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(X)

    explained = float(pca.explained_variance_ratio_.sum())
    logging.info(
        f"  done. {args.d_model} components retain {explained:.1%} of residue variance "
        f"(first PC {pca.explained_variance_ratio_[0]:.1%})"
    )

    projection_path = Path(args.projection)
    projection_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "components": torch.from_numpy(pca.components_.astype(np.float32)),
            "d_model": int(args.d_model),
            "input_dim": int(X.shape[1]),
            "explained_variance_ratio_sum": explained,
            "n_residues_fit": int(X.shape[0]),
            "n_tfs_fit": len(files),
            "sources": [str(d) for d in args.raw_dir],
            "seed": int(args.seed),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        projection_path,
    )
    logging.info(f"  saved projection -> {projection_path}")
    logging.info(
        "  KEEP THIS FILE. Any TF embedded later must be projected with it, or it lands "
        "in a different basis and is not comparable to the others."
    )


# ---------------------------------------------------------------------------
# Stage 3: apply the shared projection
# ---------------------------------------------------------------------------

def load_projection(projection_path):
    proj = torch.load(projection_path, map_location="cpu", weights_only=True)
    return proj["mean"].numpy(), proj["components"].numpy(), proj


def apply_projection(emb, mean, components):
    """[L, input_dim] -> [L, d_model]. Plain PCA transform, shared by every TF."""
    return (emb - mean) @ components.T


def run_project_stage(args):
    mean, components, meta = load_projection(args.projection)
    logging.info(
        f"Loaded projection {meta['input_dim']} -> {meta['d_model']} "
        f"(fit on {meta['n_tfs_fit']} TFs, {meta['n_residues_fit']:,} residues, "
        f"{meta['explained_variance_ratio_sum']:.1%} variance)"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = 0
    for path in _raw_files(args.raw_dir):
        tf_id = path.name.replace(RAW_SUFFIX, "")
        out_path = out_dir / f"{tf_id}{OUT_SUFFIX}"

        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        emb = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]

        projected = apply_projection(emb, mean, components)
        torch.save(torch.from_numpy(projected.astype(np.float32)), out_path)
        n_done += 1

    logging.info(f"Project stage done: {n_done} written, {n_skipped} already present. -> {out_dir}")

    if n_skipped and not args.overwrite:
        logging.warning(
            f"{n_skipped} embeddings were left in place. If any of them predate this "
            f"projection they are in a DIFFERENT basis -- rerun with --overwrite, or "
            f"delete {out_dir} first."
        )


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract TF embeddings using ProstT5 (AA + 3Di), in three stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical use (both species share one projection):\n"
            "  --stage raw     --aa_dir <mm10 fastas> --di_fasta <mm10 3di> --raw_dir <mm10 raw>\n"
            "  --stage raw     --aa_dir <hg38 fastas> --di_fasta <hg38 3di> --raw_dir <hg38 raw>\n"
            "  --stage fit     --raw_dir <mm10 raw> <hg38 raw> --projection <shared .pt>\n"
            "  --stage project --raw_dir <mm10 raw> --projection <shared .pt> --out_dir <mm10 emb>\n"
            "  --stage project --raw_dir <hg38 raw> --projection <shared .pt> --out_dir <hg38 emb>\n"
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["raw", "fit", "project"],
        help="raw: run ProstT5. fit: fit the shared PCA. project: apply it.",
    )
    parser.add_argument("--aa_dir", help="Directory with AA FASTA files (stage raw)")
    parser.add_argument(
        "--di_fasta",
        help="Foldseek 3Di FASTA file eg. ../3di_out/pdb_3Di_ss.fasta (stage raw)",
    )
    parser.add_argument(
        "--raw_dir",
        nargs="+",
        help=(
            "Raw 2048-d embedding directory. One for stage raw/project; pass every "
            "species' directory for stage fit so they share one basis."
        ),
    )
    parser.add_argument("--out_dir", help="Output directory for projected embeddings (stage project)")
    parser.add_argument(
        "--projection",
        help="Path to the shared projection .pt (written by stage fit, read by stage project)",
    )
    parser.add_argument(
        "--d_model", type=int, default=128, help="Projected embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--fit_max_residues",
        type=int,
        default=200_000,
        help="Residues to sample for the PCA fit (default: 200000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling and the PCA solver")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist instead of skipping them",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu (stage raw)")

    args = parser.parse_args()

    def require(*names):
        missing = [n for n in names if getattr(args, n) is None]
        if missing:
            parser.error(f"--stage {args.stage} requires: {', '.join('--' + m for m in missing)}")

    if args.stage == "raw":
        require("aa_dir", "di_fasta", "raw_dir")
        if len(args.raw_dir) != 1:
            parser.error("--stage raw takes exactly one --raw_dir")
        args.raw_dir = args.raw_dir[0]
        run_raw_stage(args)

    elif args.stage == "fit":
        require("raw_dir", "projection")
        run_fit_stage(args)

    elif args.stage == "project":
        require("raw_dir", "projection", "out_dir")
        run_project_stage(args)

    logging.info("\nAll done!")


if __name__ == "__main__":
    main()
