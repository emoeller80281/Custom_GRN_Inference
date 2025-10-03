from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

print(ROOT_DIR)
ORGANISM_CODE = "mm10"

DATA_DIR = ROOT_DIR / "data"
GENOME_DIR = DATA_DIR / "genome_data"
JASPAR_DIR = DATA_DIR / "motif_information"

print(DATA_DIR)