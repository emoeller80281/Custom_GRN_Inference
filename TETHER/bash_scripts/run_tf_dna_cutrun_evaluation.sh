#!/bin/bash -l
#SBATCH --job-name=tf_dna_cutrun
#SBATCH --output=LOGS/tf_dna_cutrun_%j.log
#SBATCH --error=LOGS/tf_dna_cutrun_%j.err
#SBATCH --partition=gpu,dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Submit from the repository root after running the recorded download commands:
# sbatch TETHER/bash_scripts/run_tf_dna_cutrun_evaluation.sh
# The executed notebook is saved in place, including outputs from failed runs.
set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == *hgtw* ]]; then
    echo "Submit with sbatch; this script must run on a compute node." >&2
    exit 1
fi
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
export TF_DNA_CUTRUN_BATCH_JOB_ID="$SLURM_JOB_ID"
source activate my_env
set -u
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python - <<'PY'
from pathlib import Path
import ast
import nbformat
from nbclient import NotebookClient

notebook_path = Path("TETHER/scripts/model_evaluation/tf_dna_vs_cut&run.ipynb").resolve()
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
for cell_number, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        ast.parse(cell.source, filename=f"{notebook_path.name}:cell{cell_number}")
print(f"Notebook schema and code syntax passed: {len(notebook.cells)} cells", flush=True)
client = NotebookClient(
    notebook, timeout=3600, kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parents[3])}},
)
def report_cell_start(cell, cell_index, **kwargs):
    print(f"Executing cell {cell_index}: {cell.source.splitlines()[0][:120]}", flush=True)
client.on_cell_start = report_cell_start
try:
    client.execute()
finally:
    temporary_path = notebook_path.with_suffix(".executing.ipynb")
    nbformat.write(notebook, temporary_path)
    temporary_path.replace(notebook_path)
errors = [output for cell in notebook.cells if cell.cell_type == "code"
          for output in cell.outputs if output.output_type == "error"]
assert not errors, errors
print("Notebook completed successfully; outputs saved in place.", flush=True)
PY
