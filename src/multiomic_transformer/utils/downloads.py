import os
import pandas as pd
from pathlib import Path
import logging
from typing import Union, Dict
import requests
from pybiomart import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import *

def download_gene_tss_file(
    save_dir: Union[Path, str], 
    gene_dataset_name: str = "mmusculus_gene_ensembl", 
    ensembl_version: str = "useast.ensembl.org"
    ) -> pd.DataFrame:
    """
    Downloads the gene TSS coordinates from Ensembl BioMart using pybiomart.

    Parameters
    ----------
    save_dir: Union[Path, str]
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
            "Gene name": "gene_name",
            "Chromosome/scaffold name": "chromosome",
            "Transcription start site (TSS)": "tss",
        }
    )

    # Filter out non-standard chromosomes
    df = df[df["chromosome"].str.match(r"^\d+$|^X$|^Y$")]

    # Drop duplicates
    df = df.drop_duplicates(subset=["gene_name", "chromosome", "tss"]).reset_index(drop=True)

    df["end"] = df["tss"] + 1

    df = df[["chromosome", "tss", "end", "gene_name"]]

    save_file = Path(save_dir) / "gene_tss.bed"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(save_file, sep="\t", header=False, index=False)

    return df

def download_genome_fasta(organism_code: str, save_location: Union[Path, str]):
    """
    Downloads a genome FASTA file (e.g., mm10 or hg38) from UCSC's goldenPath.

    Parameters
    ----------
    organism_code : str
        Either 'mm10' (mouse) or 'hg38' (human).
    save_location : Path or str
        Destination file path to save the .fa.gz file.
    """
    assert organism_code in ("mm10", "hg38"), \
        f"Organism code was given '{organism_code}'. Valid codes are 'mm10' or 'hg38'."

    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{organism_code}/bigZips/{organism_code}.fa.gz"

    save_path = Path(save_location)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {organism_code} genome from:\n  {url}")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {organism_code}: HTTP {response.status_code}")

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=10 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Download complete: {save_path.resolve()}")
    return save_path

def download_jaspar_pfms(save_dir: str, tax_id: str = "10090", version: int = 2024, max_workers: int = 8):
    """
    Download all TF PFM files from JASPAR REST API.

    Parameters
    ----------
    save_dir : str
        Directory to save .pfm files.
    tax_id : str
        NCBI taxonomy code used by JASPAR
        (e.g., '10090' for mouse, '9606' for human).
    version : int
        JASPAR database version (default 2024).
    max_workers : int
        Number of parallel threads for downloading.
    """
    base = f"https://jaspar.genereg.net/api/v1/matrix/?tax_id=10090&version={version}&page_size=500"

    logging.info(f"Querying JASPAR {version} mouse matrices...")
    response = requests.get(base)
    response.raise_for_status()
    data = response.json()

    results = data["results"]
    pfm_urls = {r["matrix_id"]: r["pfm"] for r in results}

    logging.info(f"Found {len(pfm_urls)} motifs for mouse.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _download(url: str, dest: Path, chunk_size: int = 1 << 18):
        """Stream download a file to destination."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            logging.info(f"  - {dest.name}")
        except Exception as e:
            logging.error(f"  - Failed to download {url}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for motif_id, url in pfm_urls.items():
            dest = save_dir / f"{motif_id}.pfm"
            if not dest.exists():
                futures.append(executor.submit(_download, url, dest))
            else:
                logging.info(f"Already exists: {dest.name}")

        for fut in as_completed(futures):
            fut.result()  # re-raise any exceptions

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
