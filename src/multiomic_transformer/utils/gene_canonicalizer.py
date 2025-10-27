import pandas as pd, numpy as np, re, unicodedata, gzip, io, os
from collections import defaultdict
from mygene import MyGeneInfo
import logging

import warnings
# silence common HTTP and library chatter
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("mygene").setLevel(logging.WARNING)

# optionally hide generic warnings (or target a specific category if you identify it)
warnings.filterwarnings("ignore")

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
    def __init__(self, species: str = "10090", use_mygene: bool = True):
        self.ens2sym = {}      # ENSMUSG000000... -> OFFICIAL_SYMBOL
        self.entrez2sym = {}   # 12345 -> OFFICIAL_SYMBOL
        self.alias2sym = {}    # ALIAS -> OFFICIAL_SYMBOL
        self.sym_ok = set()    # known official symbols
        self.species = species  # "mouse"/"human" or "10090"/"9606"
        self.use_mygene = use_mygene
        self._mg = MyGeneInfo() if use_mygene else None
        self._cache = {}

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

    def canonicalize_series(self, s: pd.Series, batch_size: int = 1000) -> pd.Series:
        """Fast path: local maps first; optional batched MyGene fallback for unresolved."""
        # 1) normalize input
        s_norm = s.astype(str).map(_norm_symbol)

        # 2) local maps
        out = s_norm.copy()

        # curated first
        out = out.map(lambda x: self.curated.get(x, x))

        # already official
        out = out.map(lambda x: x if x in self.sym_ok else x)

        # Ensembl, Entrez, alias
        def _local_map(x: str) -> str:
            if not x:
                return ""
            if x in self.sym_ok:
                return x
            if x.startswith(("ENSMUSG", "ENSG")) and x in self.ens2sym:
                return self.ens2sym[x]
            if x.isdigit() and x in self.entrez2sym:
                return self.entrez2sym[x]
            if x in self.alias2sym:
                return self.alias2sym[x]
            return x

        out = out.map(_local_map)

        # 3) MyGene fallback (batched) for anything still unmapped
        if self.use_mygene:
            unresolved = sorted({x for x in out.unique() if x and x not in self.sym_ok
                                and not x.isdigit()
                                and not x.startswith(("ENSMUSG", "ENSG"))
                                and x == _local_map(x)})
            # apply cache first
            unresolved = [u for u in unresolved if u not in self._cache]

            for i in range(0, len(unresolved), batch_size):
                chunk = unresolved[i:i+batch_size]
                try:
                    res = self._mg.querymany(
                        chunk,
                        scopes=["symbol", "alias", "ensembl.gene", "entrezgene", "uniprot"],
                        fields="symbol",
                        species=self.species,
                        verbose=False,
                        returnall=False,
                    )
                    if isinstance(res, list):
                        for r in res:
                            q = _norm_symbol(r.get("query", ""))
                            sym = _norm_symbol(r.get("symbol", "")) if not r.get("notfound") else ""
                            self._cache[q] = sym or q
                except Exception as e:
                    logging.debug(f"MyGene batch failed ({len(chunk)} items): {e}")

            # fill from cache (or identity)
            out = out.map(lambda x: self._cache.get(x, x))

        return out
    
    def canonical_symbol(self, s: str) -> str:
        s0 = _norm_symbol(s)
        if not s0:
            return ""

        # 2) curated overrides
        if s0 in self.curated:
            return self.curated[s0]

        # 3) already an official symbol
        if s0 in self.sym_ok:
            return s0

        # 4) Ensembl gene ID (strip version already handled in _norm_symbol)
        if s0.startswith(("ENSMUSG", "ENSG")):
            sym = self.ens2sym.get(s0)
            if sym:
                return sym

        # 5) Entrez numeric
        if s0.isdigit():
            sym = self.entrez2sym.get(s0)
            if sym:
                return sym

        # 6) alias
        sym = self.alias2sym.get(s0)
        if sym:
            return sym

        # 7) tiny heuristic
        if "NFKB" in s0 and s0 in self.alias2sym:
            return self.alias2sym[s0]

        # 8) optional MyGene fallback
        if self.use_mygene:
            if s0 in self._cache:
                return self._cache[s0]
            try:
                res = self._mg.querymany(
                    [s0],
                    scopes=["symbol", "alias", "ensembl.gene", "entrezgene", "uniprot"],
                    fields="symbol",
                    species=self.species,
                    verbose=False,
                    returnall=False,
                )
                sym = ""
                if res and isinstance(res, list):
                    r = res[0]
                    if not r.get("notfound") and r.get("symbol"):
                        sym = _norm_symbol(r["symbol"])
                self._cache[s0] = sym or s0  # fall back to s0 if still unknown
                return self._cache[s0]
            except Exception:
                # stay deterministic if network hiccups
                return s0

        # fallback: return normalized original
        return s0

    def standardize_df(self, df: pd.DataFrame, tf_col: str, tg_col: str) -> pd.DataFrame:
        out = df.copy()
        out[tf_col] = self.canonicalize_series(out[tf_col])
        out[tg_col] = self.canonicalize_series(out[tg_col])
        before = len(out)
        out = out[(out[tf_col] != "") & (out[tg_col] != "")]
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
