import pandas as pd, numpy as np, re, unicodedata, gzip, io, os
from collections import defaultdict

GREEK_FIX = {
    "α":"A","β":"B","γ":"G","δ":"D","ε":"E","ζ":"Z","η":"H","θ":"TH",
    "ι":"I","κ":"K","λ":"L","μ":"M","ν":"N","ξ":"X","ο":"O","π":"P",
    "ρ":"R","σ":"S","τ":"T","υ":"Y","φ":"PH","χ":"CH","ψ":"PS","ω":"O",
    "κ":"K","Κ":"K","Ω":"O","Λ":"L"
}

_MONTH2PREFIX = {
    "SEP":  "SEPT",   # SEPT1..SEPT14
    "MAR":  "MARCH",  # MARCH1..MARCH11
    # Add more only if you have clear one-to-one mappings; most others are ambiguous.
}

_date_pat = re.compile(r"^(\d{1,2})-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$", re.IGNORECASE)

def _deexcelize_symbol(sym: str) -> str:
    s = str(sym).strip()
    m = _date_pat.match(s.upper())
    if not m:
        return s
    num, mon = m.group(1), m.group(2).upper()
    if mon in _MONTH2PREFIX:
        return f"{_MONTH2PREFIX[mon]}{int(num)}"  # e.g., 1-SEP -> SEPT1, 10-SEP -> SEPT10, 1-MAR -> MARCH1
    # ambiguous months: leave unchanged
    return s


def _asciify(s: str) -> str:
    # strip accents & convert common unicode to ascii-ish
    s = "".join(GREEK_FIX.get(ch, ch) for ch in s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _norm_symbol(s: str) -> str:
    if s is None or pd.isna(s):
        return ""
    s = str(s).strip()
    s = _asciify(s)
    s = _deexcelize_symbol(s)
    s = s.replace("NF-kB","NFKB").replace("NF-kappaB","NFKB").replace("NF-kB1","NFKB1")
    s = s.replace("NF-kB2","NFKB2").replace("NF-kB p65","RELA").replace("NF-kB p50","NFKB1")
    s = s.replace(" ","").replace("\t","")
    s = s.upper()
    return s

def _split_synonyms(s: str):
    if pd.isna(s) or not str(s).strip():
        return []
    # NCBI uses | between synonyms; also split commas if present
    parts = re.split(r"[|,;/]", str(s))
    return [p.strip() for p in parts if p.strip() and p.strip() != "-"]

class GeneCanonicalizer:
    """
    Canonicalizes gene identifiers (Ensembl, Entrez, aliases) to a preferred symbol
    using local files (GTF + NCBI/MGI gene_info). Species-agnostic; pass the correct files.
    """
    def __init__(self):
        self.ens2sym = {}      # ENSMUSG000000... -> OFFICIAL_SYMBOL
        self.entrez2sym = {}   # 12345 -> OFFICIAL_SYMBOL
        self.alias2sym = {}    # ALIAS -> OFFICIAL_SYMBOL
        self.sym_ok = set()    # known official symbols

        # Optional: curated TF alias tweaks
        self.curated = {
            "HIF2A":"EPAS1", "P53":"TP53", "P73":"TRP73", "P63":"TP63",
            "NFKB":"NFKB1", "NFKB P65":"RELA", "NFKB P50":"NFKB1",
            "MYC/MAX":"MYC"
        }

    def load_gtf(self, gtf_path: str):
        # read only 'gene' rows: attributes have gene_id and gene_name
        openf = gzip.open if gtf_path.endswith(".gz") else open
        with openf(gtf_path, "rt") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 9 or cols[2] != "gene":
                    continue
                attrs = cols[8]
                m_id = re.search(r'gene_id "([^"]+)"', attrs)
                m_name = re.search(r'gene_name "([^"]+)"', attrs)
                if m_id and m_name:
                    ens = m_id.group(1)
                    sym = _norm_symbol(m_name.group(1))
                    if ens and sym:
                        self.ens2sym[ens] = sym
                        self.sym_ok.add(sym)

    def load_ncbi_gene_info(self, gene_info_path: str, species_taxid="10090"):
        """
        NCBI gene_info format:
        tax_id GeneID Symbol LocusTag Synonyms dbXrefs chromosome map_location description type_of_gene ... (tab-separated)
        """
        openf = gzip.open if gene_info_path.endswith(".gz") else open
        with openf(gene_info_path, "rt", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                tax_id, gene_id, symbol, _, syns, dbxrefs = parts[:6]
                if tax_id != species_taxid:
                    continue
                sym = _norm_symbol(symbol)
                if sym:
                    self.sym_ok.add(sym)
                    self.entrez2sym[gene_id] = sym
                    # aliases
                    for a in _split_synonyms(syns):
                        a2 = _norm_symbol(a)
                        if a2 and a2 != sym:
                            self.alias2sym.setdefault(a2, sym)
                # dbxrefs may contain Ensembl:ENSMUSG...
                for x in _split_synonyms(dbxrefs):
                    if x.upper().startswith("ENSEMBL:"):
                        ens = x.split(":",1)[1].strip()
                        if ens:
                            self.ens2sym[ens] = sym

    def canonical_symbol(self, s: str) -> str:
        s0 = _norm_symbol(s)
        if not s0:
            return ""
        # curated overrides first
        if s0 in self.curated:
            return self.curated[s0]
        # already an official symbol?
        if s0 in self.sym_ok:
            return s0
        # Ensembl gene ID?
        if s0.startswith("ENSMUSG") or s0.startswith("ENSG"):
            return self.ens2sym.get(s0, s0)  # if unknown, leave as is (will likely get dropped)
        # Entrez numeric?
        if s0.isdigit():
            return self.entrez2sym.get(s0, s0)
        # alias?
        if s0 in self.alias2sym:
            return self.alias2sym[s0]
        # try small heuristics for NFkB spelling
        if "NFKB" in s0 and s0 not in self.sym_ok and s0 in self.alias2sym:
            return self.alias2sym[s0]
        return s0  # fallback

    def standardize_df(self, df: pd.DataFrame, tf_col: str, tg_col: str) -> pd.DataFrame:
        out = df.copy()
        out[tf_col] = out[tf_col].apply(self.canonical_symbol)
        out[tg_col] = out[tg_col].apply(self.canonical_symbol)
        # drop empties
        before = len(out)
        out = out[(out[tf_col].astype(str) != "") & (out[tg_col].astype(str) != "")]
        dropped = before - len(out)
        if dropped:
            print(f"[Canonicalizer] Dropped {dropped} rows with empty/unmappable TF/TG")
        return out

    def coverage_report(self):
        return {
            "n_official": len(self.sym_ok),
            "n_ens2sym": len(self.ens2sym),
            "n_entrez2sym": len(self.entrez2sym),
            "n_alias2sym": len(self.alias2sym),
        }
