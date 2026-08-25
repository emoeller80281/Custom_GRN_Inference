"""Locate each TF's DNA-binding domain in its embedded protein sequence.

Writes one row per (TF, domain) with residue coordinates that index directly into
the per-residue TF embeddings, so downstream code can pool over the DBD instead of
the whole protein.

Two annotation sources, both over REST -- no local InterProScan install, which would
need Java plus ~50 GB of member-database files:

  * UniProt curated features -- the ``DNA binding`` and ``Zinc finger`` feature keys.
    Precise and hand-curated, but only on reviewed (Swiss-Prot) entries.
  * InterPro Pfam match locations -- broader coverage. Which Pfam domains count as the
    DBD is not guessed: CIS-BP's ``DBDs`` column already names them in Pfam short-name
    vocabulary (``zf-C2H2``, ``bZIP_1``, ``HMG_box``, ...). A Pfam match is marked
    ``is_dbd`` when its short name is in that TF's CIS-BP list *or* when it overlaps a
    UniProt curated DNA-binding feature. The second rule matters: Pfam renames families
    between releases (PF00046 went ``Homeobox`` -> ``Homeodomain``), and CIS-BP still
    carries the old vocabulary, so a pure string match silently drops homeodomains --
    the single largest DBD family here.

Coordinates are 1-based inclusive, on the *local FASTA* sequence -- the one that was
embedded. UniProt is a different database with its own sequence, so every mapping is
length-checked before its coordinates are trusted; ``seq_status`` records the outcome
and rows that could not be verified are marked rather than silently emitted.

Usage:
    python3 scripts/fetch_tf_domains.py --species mm10
    python3 scripts/fetch_tf_domains.py --species hg38 --force_reload
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import config  # noqa: E402

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
INTERPRO_PFAM = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{acc}/"
INTERPRO_ENTRY = "https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam}/"

TAXON_ID = {"mm10": 10090, "hg38": 9606}
UNIPROT_BATCH = 50
DBD_FEATURES = ("DNA binding", "Zinc finger")


def polite_get(session, url, params=None, tries=4, pause=0.4):
    """GET with backoff. EBI and UniProt both throttle rather than hard-fail."""
    for attempt in range(tries):
        response = session.get(url, params=params, timeout=60)
        if response.status_code == 200:
            time.sleep(pause)
            return response
        if response.status_code in (204, 404):
            # 204 = the protein simply has no matches; not worth retrying.
            return None
        time.sleep(pause * 2 ** attempt)
    logging.warning("  giving up on %s (HTTP %s)", url, response.status_code)
    return None


def read_local_proteins(sequence_dir):
    """{tf_name: (refseq_accession, sequence)} from the FASTAs that were embedded."""
    proteins = {}
    for path in sorted(Path(sequence_dir).glob("*_protein.fasta")):
        record = next(SeqIO.parse(path, "fasta"), None)
        if record is None:
            continue
        proteins[path.name.replace("_protein.fasta", "")] = (record.id, str(record.seq))
    return proteins


def map_refseq_to_uniprot(session, proteins, taxon_id):
    """RefSeq NP_* -> UniProt, batched, reviewed entries preferred.

    A RefSeq accession can hit several UniProt entries (isoforms, plus unreviewed
    TrEMBL duplicates), so reviewed entries are queried first and unreviewed used
    only to fill the gaps.
    """
    fields = "accession,reviewed,xref_refseq,sequence,ft_dna_bind,ft_zn_fing,xref_pfam"
    by_refseq = {}

    for reviewed in (True, False):
        pending = [
            refseq for _, (refseq, _) in proteins.items() if refseq not in by_refseq
        ]
        if not pending:
            break
        logging.info(
            "  UniProt lookup (%s): %d accessions",
            "reviewed" if reviewed else "unreviewed",
            len(pending),
        )
        for start in range(0, len(pending), UNIPROT_BATCH):
            chunk = pending[start : start + UNIPROT_BATCH]
            query = "(" + " OR ".join(f"xref:{a}" for a in chunk) + ")"
            query += f" AND organism_id:{taxon_id}"
            if reviewed:
                query += " AND reviewed:true"
            response = polite_get(
                session,
                UNIPROT_SEARCH,
                {"query": query, "fields": fields, "format": "json", "size": "500"},
            )
            if response is None:
                continue
            for entry in response.json().get("results", []):
                # Link the entry back to whichever queried accession it cites.
                cited = {
                    x["id"].split(".")[0]
                    for x in entry.get("uniProtKBCrossReferences", [])
                    if x["database"] == "RefSeq"
                }
                for refseq in chunk:
                    if refseq.split(".")[0] in cited and refseq not in by_refseq:
                        by_refseq[refseq] = entry
    return by_refseq


def uniprot_domain_rows(tf_name, entry, local_seq, seq_status):
    """Curated ``DNA binding`` / ``Zinc finger`` features."""
    rows = []
    for feature in entry.get("features", []):
        if feature["type"] not in DBD_FEATURES:
            continue
        location = feature["location"]
        rows.append(
            dict(
                tf_name=tf_name,
                uniprot=entry["primaryAccession"],
                source="uniprot_feature",
                domain=feature.get("description") or feature["type"],
                pfam="",
                start=location["start"]["value"],
                end=location["end"]["value"],
                is_dbd=True,
                seq_status=seq_status,
                protein_length=len(local_seq),
            )
        )
    return rows


def pfam_short_names(session, entry):
    """{PFxxxxx: short name} straight off the UniProt cross-references."""
    names = {}
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref["database"] != "Pfam":
            continue
        for prop in xref.get("properties", []):
            if prop["key"] == "EntryName":
                names[xref["id"]] = prop["value"]
    return names


def _overlaps_curated(start, end, curated):
    """True if [start, end] covers >= half of some curated DNA-binding interval."""
    for c_start, c_end in curated:
        shared = min(end, c_end) - max(start, c_start) + 1
        if shared > 0 and shared >= 0.5 * min(end - start + 1, c_end - c_start + 1):
            return True
    return False


def interpro_domain_rows(
    session, tf_name, entry, local_seq, seq_status, dbd_names, short_name_cache,
    curated_intervals, interpro_cache,
):
    """Pfam match coordinates from InterPro, flagged against the CIS-BP DBD list."""
    accession = entry["primaryAccession"]
    if accession in interpro_cache:
        payload = interpro_cache[accession]
    else:
        response = polite_get(session, INTERPRO_PFAM.format(acc=accession), {"page_size": 200})
        payload = response.json() if response is not None else {"results": []}
        interpro_cache[accession] = payload

    from_uniprot = pfam_short_names(session, entry)
    rows = []
    for result in payload.get("results", []):
        pfam = result["metadata"]["accession"]

        short = from_uniprot.get(pfam) or short_name_cache.get(pfam)
        if short is None:
            detail = polite_get(session, INTERPRO_ENTRY.format(pfam=pfam))
            name = detail.json()["metadata"]["name"] if detail else None
            short = name.get("short", "") if isinstance(name, dict) else ""
            short_name_cache[pfam] = short

        for protein in result["proteins"]:
            for location in protein["entry_protein_locations"]:
                for fragment in location["fragments"]:
                    rows.append(
                        dict(
                            tf_name=tf_name,
                            uniprot=accession,
                            source="interpro_pfam",
                            domain=short,
                            pfam=pfam,
                            start=fragment["start"],
                            end=fragment["end"],
                            is_dbd=(
                                short in dbd_names
                                or _overlaps_curated(
                                    fragment["start"], fragment["end"], curated_intervals
                                )
                            ),
                            seq_status=seq_status,
                            protein_length=len(local_seq),
                        )
                    )
    return rows


def cisbp_dbd_names(motif_dir):
    """{TF_NAME: {Pfam short names that CIS-BP calls this TF's DBD}}."""
    info = pd.read_csv(Path(motif_dir) / "TF_Information_all_motifs.txt", sep="\t")
    info["TF_Name"] = info["TF_Name"].str.upper()
    out = {}
    for tf_name, rows in info.groupby("TF_Name"):
        names = set()
        for value in rows["DBDs"].dropna().unique():
            names.update(part.strip() for part in str(value).split(",") if part.strip())
        out[tf_name] = names
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True, choices=sorted(TAXON_ID))
    parser.add_argument("--out", default=None, help="output CSV (default: data/tf_data/<species>/tf_domains.csv)")
    parser.add_argument("--force_reload", action="store_true", help="ignore the raw-response cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tf_data_dir = PROJECT_DIR.parent / "data" / "tf_data" / args.species
    motif_dir = PROJECT_DIR.parent / "data" / "databases" / "motif_information" / args.species
    out_path = Path(args.out) if args.out else tf_data_dir / "tf_domains.csv"
    cache_path = tf_data_dir / "uniprot_entries.json"
    interpro_cache_path = tf_data_dir / "interpro_pfam.json"

    proteins = read_local_proteins(tf_data_dir / "tf_sequences")
    logging.info("Read %d local protein FASTAs", len(proteins))

    session = requests.Session()
    session.headers.update({"User-Agent": "TETHER-domain-fetch"})

    if cache_path.exists() and not args.force_reload:
        logging.info("Loading cached UniProt entries from %s", cache_path)
        by_refseq = json.loads(cache_path.read_text())
    else:
        by_refseq = map_refseq_to_uniprot(session, proteins, TAXON_ID[args.species])
        cache_path.write_text(json.dumps(by_refseq))
    logging.info("Mapped %d/%d TFs to UniProt", len(by_refseq), len(proteins))

    dbd_lookup = cisbp_dbd_names(motif_dir)
    interpro_cache = (
        json.loads(interpro_cache_path.read_text())
        if interpro_cache_path.exists() and not args.force_reload
        else {}
    )
    short_name_cache = {}
    rows, unmapped = [], []

    for i, (tf_name, (refseq, local_seq)) in enumerate(sorted(proteins.items()), 1):
        entry = by_refseq.get(refseq)
        if entry is None:
            unmapped.append(tf_name)
            continue

        # UniProt is a separate database with its own sequence. Its coordinates are
        # only usable if that sequence lines up with the one that was embedded.
        uniprot_seq = entry.get("sequence", {}).get("value", "")
        if uniprot_seq == local_seq:
            seq_status = "exact"
        elif len(uniprot_seq) == len(local_seq):
            seq_status = "same_length"
        else:
            seq_status = "length_mismatch"

        if seq_status == "length_mismatch":
            unmapped.append(tf_name)
            continue

        dbd_names = dbd_lookup.get(tf_name.upper(), set())
        curated = uniprot_domain_rows(tf_name, entry, local_seq, seq_status)
        rows += curated
        rows += interpro_domain_rows(
            session, tf_name, entry, local_seq, seq_status, dbd_names, short_name_cache,
            [(r["start"], r["end"]) for r in curated], interpro_cache,
        )

        if i % 50 == 0:
            logging.info("  %d/%d TFs, %d domain rows", i, len(proteins), len(rows))

    interpro_cache_path.write_text(json.dumps(interpro_cache))
    domains = pd.DataFrame(rows)
    domains.to_csv(out_path, index=False)

    logging.info("\nWrote %d domain rows to %s", len(domains), out_path)
    if len(domains):
        with_dbd = domains.loc[domains.is_dbd, "tf_name"].nunique()
        logging.info("  TFs with >=1 DBD located: %d/%d", with_dbd, len(proteins))
        logging.info("  by source:\n%s", domains.groupby(["source", "is_dbd"]).size())
        logging.info("  sequence check:\n%s", domains.drop_duplicates("tf_name").seq_status.value_counts())
    if unmapped:
        logging.info("  unmapped/unverifiable (%d): %s", len(unmapped), unmapped[:15])


if __name__ == "__main__":
    main()
