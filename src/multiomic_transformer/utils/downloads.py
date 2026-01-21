import os
import pandas as pd
import subprocess
from pathlib import Path
import logging
import shutil
from typing import Union, Dict, Tuple, Optional
import requests
import gzip
import pysam
from tqdm.auto import tqdm
from pybiomart import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_gene_tss_file(
    save_file: Union[Path, str], 
    gene_dataset_name: str = "hsapiens_gene_ensembl", 
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

def download_genome_fasta(organism_code: str, save_dir: Union[str, Path]) -> Path:
    """
    Download a UCSC genome FASTA, overwrite gzip with BGZF (still .gz), and index via pysam.faidx.
    """
    assert organism_code in ("mm10", "hg38"), \
        f"Organism code '{organism_code}' not supported (valid: 'mm10', 'hg38')."

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gz_path  = save_dir / f"{organism_code}.fa.gz"      # final file (BGZF) keeps .gz suffix
    fai_path = save_dir / f"{organism_code}.fa.gz.fai"  # index alongside .gz
    gzi_path = save_dir / f"{organism_code}.fa.gz.gzi"  # BGZF index
    
    def _download_with_progress(url: str, dest: Path, chunk_size: int = 256 * 1024, desc: str = "") -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with requests.get(url, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=desc or dest.name,
                dynamic_ncols=True, mininterval=0.1,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp.replace(dest)

    # 1) Download gzip if missing
    if not gz_path.exists():
        url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{organism_code}/bigZips/{organism_code}.fa.gz"
        logging.info(f"Downloading {organism_code} genome from:\n  {url}")
        _download_with_progress(url, gz_path, desc=gz_path.name)
        logging.info(f"  - Download complete: {gz_path}")
    else:
        logging.info(f"  - Found existing genome file: {gz_path}")
        
    def _is_bgzf(path: Path) -> bool:
        """
        True iff file is BGZF (gzip with 'BC' extra subfield).
        Fast path: if a .gzi sibling exists (non-empty), assume BGZF.
        """
        try:
            gzi = Path(str(path) + ".gzi")  # e.g., mm10.fa.gz.gzi
            if gzi.exists() and gzi.stat().st_size > 0:
                return True

            with open(path, "rb") as fh:
                hdr = fh.read(10)  # ID1 ID2 CM FLG MTIME(4) XFL OS  -> 10 bytes
                if len(hdr) < 10 or hdr[0:2] != b"\x1f\x8b" or hdr[2] != 8:
                    return False
                flg = hdr[3]
                if not (flg & 0x04):  # FEXTRA not set
                    return False

                # Read XLEN (2 bytes, little-endian), then that many bytes of extra
                xlen_b = fh.read(2)
                if len(xlen_b) != 2:
                    return False
                xlen = int.from_bytes(xlen_b, "little")
                extra = fh.read(xlen)

                # Walk subfields looking for 'BC'
                i = 0
                while i + 4 <= len(extra):
                    si = extra[i:i+2]
                    slen = int.from_bytes(extra[i+2:i+4], "little")
                    i += 4
                    if si == b"BC":
                        return True
                    i += slen
                return False
        except Exception:
            return False

    # 2) Ensure it's BGZF; UCSC provides plain gzip, so this will usually run once
    if not _is_bgzf(gz_path):
        logging.info(f"{gz_path.name} is plain gzip; converting to BGZF (keeping .gz)…")
        tmp_bgzf = gz_path.with_suffix(gz_path.suffix + ".bgzf.tmp")
        tmp_fa = gz_path.with_suffix("")  # uncompressed intermediate
        
        # Try using bgzip CLI (much faster, parallel) if available
        try:
            # Decompress to temp file
            logging.info(f"  - Decompressing with gunzip...")
            subprocess.run(["gunzip", "-c", str(gz_path)], 
                          stdout=open(tmp_fa, "wb"), 
                          check=True, 
                          stderr=subprocess.DEVNULL)
            
            # Compress with bgzip (parallel, much faster)
            logging.info(f"  - Compressing with bgzip...")
            subprocess.run(["bgzip", "-f", "-@", "8", str(tmp_fa)], 
                          check=True, 
                          stderr=subprocess.DEVNULL)
            
            # bgzip creates .gz file, rename to our temp name then to final
            (tmp_fa.parent / f"{tmp_fa.name}.gz").replace(tmp_bgzf)
            tmp_bgzf.replace(gz_path)
            logging.info(f"  - Converted to BGZF using bgzip")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to Python if bgzip not available
            logging.info(f"  - bgzip not available, using slower Python conversion...")
            if tmp_fa.exists():
                tmp_fa.unlink()
            
            # Use larger chunks (4MB) for better performance
            with gzip.open(gz_path, "rb") as fin, pysam.BGZFile(str(tmp_bgzf), "wb") as bout:
                chunk_size = 1 << 22  # 4MB chunks
                file_size = gz_path.stat().st_size
                bytes_read = 0
                last_log_pct = -1
                
                for chunk in iter(lambda: fin.read(chunk_size), b""):
                    if not chunk:
                        break
                    bout.write(chunk)
                    bytes_read += len(chunk)
                    
                    # Log progress every 10%
                    pct = int((bytes_read / file_size) * 100)
                    if pct >= last_log_pct + 10:
                        logging.info(f"    {pct}% complete")
                        last_log_pct = pct
            
            tmp_bgzf.replace(gz_path)
            logging.info(f"  - Conversion complete")
    else:
        logging.info(f"  - {gz_path.name} is already BGZF; skipping transcode")

    # 3) Index the BGZF FASTA (.fai + .gzi)
    if fai_path.exists() and gzi_path.exists():
        logging.info(f"  - Index already exists: {fai_path.name}, {gzi_path.name}")
    else:
        logging.info(f"Indexing {gz_path.name} with pysam.faidx …")
        pysam.faidx(str(gz_path))
        logging.info(f"  - Index created: {fai_path.name} (and {gzi_path.name})")

    logging.info(f"Genome ready: {gz_path}")
    return gz_path

def download_chrom_sizes(organism_code: str, save_dir: Union[str, Path]) -> Path:
    """
    Download UCSC chrom.sizes for an assembly (mm10 or hg38).

    Returns
    -------
    Path
        Path to the chrom.sizes file.
    """
    assert organism_code in ("mm10", "hg38"), \
        f"Organism code '{organism_code}' not supported (valid: 'mm10', 'hg38')."

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{organism_code}/bigZips/{organism_code}.chrom.sizes"
    out_path = save_dir / f"{organism_code}.chrom.sizes"

    if out_path.exists():
        logging.info(f"Found existing chrom.sizes: {out_path}")
        return out_path

    logging.info(f"Downloading chrom.sizes:\n  {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)

    logging.info(f"chrom.sizes saved: {out_path}")
    return out_path

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
    os.makedirs(save_dir, exist_ok=True)

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

    def _download(url: str, dest: str, chunk_size: int = 1 << 18):
        """Stream a file to disk safely."""
        dest = Path(dest)
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
            dest = os.path.join(save_dir, f"{matrix_id}.pfm")
            if not os.path.exists(dest):
                futures.append(executor.submit(_download, pfm_url, dest))
            else:
                logging.info(f"Already exists: {dest}")
        for fut in as_completed(futures):
            fut.result()

    logging.info(f"All PFMs saved under {save_dir}")

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

# --- tiny shared helper -------------------------------------------------------
def _stream_download(url: str, dest: Path, chunk: int = 1 << 20, desc: Optional[str] = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=desc or dest.name, dynamic_ncols=True, mininterval=0.1
        ) as pbar:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if not chunk_bytes:
                    continue
                f.write(chunk_bytes)
                pbar.update(len(chunk_bytes))
    tmp.replace(dest)

def download_ncbi_gene_info(organism_code: str, out_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Download NCBI Gene Info for mouse or human.
    
    Parameters
    ----------
    organism_code : str
        'mm10' for mouse or 'hg38' for human
    out_path : Optional[Union[str, Path]]
        Custom output path. If None, uses default based on organism.
    
    Returns
    -------
    Path
        Path to the downloaded gene_info.gz file
    """
    # Map organism code to NCBI species and paths
    org_map = {
        "mm10": {
            "species": "Mus_musculus",
            "subdir": "Mammalia",
            "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz"
        },
        "hg38": {
            "species": "Homo_sapiens",
            "subdir": "Mammalia",
            "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
        }
    }
    
    if organism_code not in org_map:
        raise ValueError(f"Unsupported organism_code: {organism_code}. Use 'mm10' or 'hg38'.")
    
    info = org_map[organism_code]
    url = info["url"]
    filename = f"{info['species']}.gene_info.gz"
    
    # Construct organism-specific path
    if out_path is not None:
        dest = Path(out_path)
    else:
        # Construct path relative to current working directory
        base = Path("data/genome_data/genome_annotation") / organism_code
        dest = base / filename
        # Make it absolute
        if not dest.is_absolute():
            dest = Path.cwd() / dest

    if dest.exists():
        logging.info(f"Found existing gene_info: {dest}")
        return dest

    logging.info(f"Downloading NCBI Gene Info for {info['species']}:\n  {url}")
    _stream_download(url, dest, desc=dest.name)
    logging.info(f"Saved: {dest}")
    return dest

# Backward compatibility
def download_ncbi_gene_info_mouse(out_path: Optional[Union[str, Path]] = None) -> Path:
    """Legacy wrapper for download_ncbi_gene_info with organism_code='mm10'"""
    return download_ncbi_gene_info("mm10", out_path)

def download_ensembl_gtf(
    organism_code: str,
    release: Optional[int] = None,
    assembly: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
    decompress: bool = False,
) -> Path:
    """
    Download Ensembl GTF for mouse or human. By default keeps .gtf.gz.
    If decompress=True, also writes an uncompressed .gtf alongside.
    
    Parameters
    ----------
    organism_code : str
        'mm10' for mouse or 'hg38' for human
    release : Optional[int]
        Ensembl release version. If None, uses defaults (115 for mouse, 113 for human)
    assembly : Optional[str]
        Genome assembly. If None, uses defaults (GRCm39 for mouse, GRCh38 for human)
    out_dir : Optional[Union[str, Path]]
        Output directory. If None, uses GTF_FILE_DIR or creates default path
    decompress : bool
        Whether to also create an uncompressed .gtf file
    
    Returns
    -------
    Path
        Path to the GTF file (.gtf.gz or .gtf if decompress=True)
    """
    # Map organism code to Ensembl parameters
    org_map = {
        "mm10": {
            "species_dir": "mus_musculus",
            "species_name": "Mus_musculus",
            "default_assembly": "GRCm39",
            "default_release": 115
        },
        "hg38": {
            "species_dir": "homo_sapiens",
            "species_name": "Homo_sapiens",
            "default_assembly": "GRCh38",
            "default_release": 113
        }
    }
    
    if organism_code not in org_map:
        raise ValueError(f"Unsupported organism_code: {organism_code}. Use 'mm10' or 'hg38'.")
    
    info = org_map[organism_code]
    release = release if release is not None else info["default_release"]
    assembly = assembly if assembly is not None else info["default_assembly"]
    
    org = info["species_dir"]
    species_name = info["species_name"]
    fn_gz = f"{species_name}.{assembly}.{release}.gtf.gz"
    url = f"https://ftp.ensembl.org/pub/release-{release}/gtf/{org}/{fn_gz}"

    # Construct organism-specific path
    if out_dir is None:
        # Construct path relative to current working directory
        out_dir = Path("data/genome_data/genome_annotation") / organism_code
        # Make it absolute
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_gz = out_dir / fn_gz
    dest_gtf = dest_gz.with_suffix("")  # drop .gz -> .gtf

    if dest_gz.exists():
        logging.info(f"Found existing GTF: {dest_gz}")
    else:
        logging.info(f"Downloading Ensembl GTF for {species_name}:\n  {url}")
        _stream_download(url, dest_gz, desc=dest_gz.name)
        logging.info(f"Saved: {dest_gz}")

    if decompress:
        if dest_gtf.exists():
            logging.info(f"Uncompressed GTF already exists: {dest_gtf}")
        else:
            logging.info(f"Decompressing → {dest_gtf.name}")
            with gzip.open(dest_gz, "rb") as fin, open(dest_gtf, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 20)
            logging.info(f"Wrote: {dest_gtf}")

    return dest_gtf if decompress else dest_gz

# Backward compatibility
def download_ensembl_gtf_mouse(
    release: int = 115,
    assembly: str = "GRCm39",
    out_dir: Optional[Union[str, Path]] = None,
    decompress: bool = False,
) -> Path:
    """Legacy wrapper for download_ensembl_gtf with organism_code='mm10'"""
    return download_ensembl_gtf("mm10", release, assembly, out_dir, decompress)