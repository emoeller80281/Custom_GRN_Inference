import os
import pandas as pd
import subprocess
from pathlib import Path
import logging
from typing import Union, Dict
import requests
from pybiomart import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import *

def download_gene_tss_file(
    save_file: Union[Path, str], 
    gene_dataset_name: str = "mmusculus_gene_ensembl", 
    ensembl_version: str = "useast.ensembl.org"
    ) -> pd.DataFrame:
    """
    Downloads the gene TSS coordinates from Ensembl BioMart using pybiomart.

    Parameters
    ----------
    save_file: Union[Path, str]
        Path to save the gene TSS bed file.
        
    ensembl_version : str, optional
        Ensembl host mirror to query (default: "useast.ensembl.org").
        Examples: "www.ensembl.org", "uswest.ensembl.org", etc.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: ['gene_name', 'chromosome', 'tss']
    """

    # Connect to the Ensembl BioMart gene dataset for the organism
    dataset = Dataset(name=gene_dataset_name, host=f"http://{ensembl_version}")

    # Retrieve TSS, gene name, and chromosome
    df = dataset.query(
        attributes=[
            "external_gene_name",
            "chromosome_name",
            "transcription_start_site",
        ]
    )

    # Clean up and rename
    df = df.rename(
        columns={
            "Gene name": "name",
            "Chromosome/scaffold name": "chrom",
            "Transcription start site (TSS)": "start",
        }
    )

    # Filter out non-standard chromosomes
    df = df[df["chrom"].str.match(r"^\d+$|^X$|^Y$")]

    # Add the chr prefix to the chromosomes
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].apply(lambda c: f"chr{c}" if not c.startswith("chr") else c)

    # Drop duplicates
    df = df.drop_duplicates(subset=["name", "chrom", "start"]).reset_index(drop=True)

    df["end"] = df["start"] + 1

    df = df.dropna()


    df = df[["chrom", "start", "end", "name"]]

    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(save_file, sep="\t", header=False, index=False)

    return df

def download_genome_fasta(organism_code: str, save_dir: Union[str, Path]):
    """
    Download a UCSC genome FASTA, convert to BGZF format, and index with samtools.

    Parameters
    ----------
    organism_code : str
        Genome assembly (e.g., 'mm10', 'hg38').
    save_dir : str or Path
        Directory to save the genome files.
    """
    assert organism_code in ("mm10", "hg38"), \
        f"Organism code '{organism_code}' not supported (valid: 'mm10', 'hg38')"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gz_path = save_dir / f"{organism_code}.fa.gz"
    bgzf_path = save_dir / f"{organism_code}.fa.bgz"
    fasta_path = save_dir / f"{organism_code}.fa"
    fai_path = save_dir / f"{organism_code}.fa.bgz.fai"

    # Step 1. Download genome if not present
    if not gz_path.exists():
        url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{organism_code}/bigZips/{organism_code}.fa.gz"
        logging.info(f"Downloading {organism_code} genome from:\n  {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        logging.info(f"  - Download complete: {gz_path}")
    else:
        logging.info(f"  - Found existing genome file: {gz_path}")

    def _run_cmd(cmd: list[str]):
        """Run a shell command and raise an error if it fails."""
        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
        return result

    # Step 2. Convert gzip → BGZF (via samtools)
    if not bgzf_path.exists():
        logging.info(f"Converting {gz_path.name} to BGZF format...")
        _run_cmd(["gunzip", "-c", str(gz_path), "|", "bgzip", ">", str(bgzf_path)])
        # Some environments may not interpret pipes; safer to split into two steps:
        # 1. gunzip -> .fa
        # 2. bgzip -> .fa.bgz
        if not bgzf_path.exists():
            _run_cmd(["gunzip", "-c", str(gz_path), ">", str(fasta_path)])
            _run_cmd(["bgzip", "-f", str(fasta_path)])
        logging.info(f"Converted to BGZF: {bgzf_path}")
    else:
        logging.info(f"Found existing BGZF file: {bgzf_path}")

    # Step 3. Index with samtools
    if not fai_path.exists():
        logging.info(f"Indexing {bgzf_path.name} with samtools...")
        _run_cmd(["samtools", "faidx", str(bgzf_path)])
        logging.info(f"Index created: {fai_path}")
    else:
        logging.info(f"Index already exists: {fai_path}")

    logging.info(f"Genome downloaded and converted to bgzf: {bgzf_path}")
    return bgzf_path

def download_jaspar_pfms(save_dir: str, tax_id: str = "10090", version: int = 2024, max_workers: int = 8):
    """
    Download all JASPAR PFMs for a given organism (e.g., mouse) via REST API.

    Parameters
    ----------
    save_dir : str
        Directory to save .pfm files.
    tax_id : str
        NCBI taxonomy ID (e.g., '10090' for mouse, '9606' for human).
    version : int
        JASPAR release version.
    max_workers : int
        Parallel download threads.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # List endpoint for this organism
    list_url = f"https://jaspar.elixir.no/api/v1/matrix/?tax_id={tax_id}&version={version}&page_size=500"
    logging.info(f"Fetching JASPAR matrices: {list_url}")

    resp = requests.get(list_url)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    logging.info(f"Found {len(results)} motifs for tax_id={tax_id}")

    # Build URLs for PFMs
    pfm_urls = {
        r["matrix_id"]: f"{r['url']}?format=pfm"
        for r in results if "url" in r
    }

    logging.info(f"Preparing to download {len(pfm_urls)} PFMs...")

    def _download(url: str, dest: Path, chunk_size: int = 1 << 18):
        """Stream a file to disk safely."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        tmp.replace(dest)
        logging.info(f"Downloaded {dest.name}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for matrix_id, pfm_url in pfm_urls.items():
            dest = save_dir / f"{matrix_id}.pfm"
            if not dest.exists():
                futures.append(executor.submit(_download, pfm_url, dest))
            else:
                logging.info(f"Already exists: {dest.name}")
        for fut in as_completed(futures):
            fut.result()

    logging.info(f"All PFMs saved under {save_dir.resolve()}")


def ensure_string_v12_files(string_dir: str, string_org_code: str) -> Dict[str, str]:
    """
    Ensure STRING v12.0 files exist locally; download if missing.

    Parameters
    ----------
    string_dir : str
        Directory to store STRING files.
    string_org_code : str
        NCBI taxonomy code used by STRING
        (e.g., '10090' for mouse, '9606' for human).

    Returns
    -------
    dict
        {
            'protein_info_gz',
            'protein_links_detailed_gz',
            'protein_info_url',
            'protein_links_detailed_url',
        }
    """
    base = "https://stringdb-downloads.org/download"
    Path(string_dir).mkdir(parents=True, exist_ok=True)

    files = {
        "protein_info_gz": f"{string_org_code}.protein.info.v12.0.txt.gz",
        "protein_links_detailed_gz": f"{string_org_code}.protein.links.detailed.v12.0.txt.gz",
    }

    urls = {
        "protein_info_url": f"{base}/protein.info.v12.0/{files['protein_info_gz']}",
        "protein_links_detailed_url": f"{base}/protein.links.detailed.v12.0/{files['protein_links_detailed_gz']}",
    }

    paths = {k: os.path.join(string_dir, v) for k, v in files.items()}

    def _download(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
        """
        Stream-download a file from `url` to `dest_path` safely.

        - Creates directories as needed
        - Writes to a temporary file first, then renames atomically
        - Logs progress
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        logging.info(f"⬇  Downloading {url} → {dest_path}")

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(dest)
        logging.info(f"Download complete: {dest.resolve()}")

    # Download if missing
    for file_key, url_key in zip(files.keys(), urls.keys()):
        path = paths[file_key]
        url = urls[url_key]
        if not os.path.exists(path):
            _download(url, path)
        else:
            logging.info(f"Found existing: {path}")

    return {**paths, **urls}
