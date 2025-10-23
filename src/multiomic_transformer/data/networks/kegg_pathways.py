import sys

sys.path = [p if isinstance(p, str) else str(p) for p in sys.path]

from bioservices.kegg import KEGG
import networkx as nx
import requests
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Union
from bs4 import BeautifulSoup
import itertools
import logging
import os
from tqdm import tqdm
from config.settings import *
import pickle
import time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.client import RemoteDisconnected
import random

class Pathways:
    """
    Reads in and processes the KEGG pathways as networkx graphs
    """
    def __init__(self, dataset_name, output_path, organism):
        self.pathway_graphs = {}
        self.dataset_name = dataset_name
        self.output_path = os.path.join(output_path, "graphml_files/")
        self.gene_indices = []
        self.pathway_dict = {}
        self.organism = organism
        self.file_paths = {"pathway_xml_files": output_path}
        self.pathway_list = self._create_global_organism_pathway_list()

        os.makedirs(self.output_path, exist_ok=True)
        
    def _create_global_organism_pathway_list(self):
        
        def silent_kegg_initialization():
            """Initializes KEGG quietly"""
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = open('/dev/null', 'w')
                sys.stderr = open('/dev/null', 'w')
                kegg = KEGG(verbose=False) 
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            return kegg
        
        k = silent_kegg_initialization()  # read KEGG from bioservices
        k.organism = self.organism
        try:
            response = requests.get(f"https://rest.kegg.jp/list/pathway/{self.organism}", timeout=30)
            response.raise_for_status()
            # KEGG returns lines like: "path:hsa00010\tGlycolysis / Gluconeogenesis"
            pathway_list = [line.split("\t")[0].replace("path:", "") for line in response.text.strip().split("\n")]
            return pathway_list
        except Exception as e:
            raise RuntimeError(f"Failed to fetch KEGG pathway list for {self.organism}: {e}")

        
    # Function to silence KEGG initialization to the terminal

    
    def _count_generator(self, reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    def parse_kegg_dict(self):
        """
        Makes a dictionary to convert ko numbers from KEGG into real gene names
        """
        logging.info(f'\t\tParsing KEGG dict...')
        gene_dict = {}

        # If the dictionary file exists, use that (much faster than streaming)
        if 'kegg_dict.csv' in os.listdir(f'{self.file_paths["pathway_xml_files"]}'):
            logging.info(f'\t\t\tReading in KEGG dictionary file...')
            with open(f'{self.file_paths["pathway_xml_files"]}/kegg_dict.csv', 'r') as kegg_dict_file:
                for line in kegg_dict_file:
                    line = line.strip().split('\t')
                    kegg_code = line[0]
                    gene_number = line[1]
                    gene_dict[kegg_code] = gene_number

        # If the dictionary file does not exist, write it and stream in the data for the dictionary
        else:
            logging.info(f'\t\t\tKEGG dictionary not found, downloading...')

            pathway_file = requests.get("https://rest.kegg.jp/get/br:ko00001", stream=True, timeout=60)
            with open(f'{self.file_paths["pathway_xml_files"]}/kegg_dict.csv', 'w') as kegg_dict_file:
                for line in pathway_file.iter_lines():
                    line = line.decode("utf-8")
                    if len(line) > 1 and line[0] == "D":  # lines which begin with D translate kegg codes to gene names
                        
                        # to split into kegg code, gene names
                        converter = re.split(r"\s+", re.split(r";", line)[0])
                        kegg_code = converter[1].upper()
                        gene_number = converter[2].upper().replace(',', '')
                        gene_dict[kegg_code] = gene_number
                        kegg_dict_file.write(f'{kegg_code}\t{gene_number}\n')
            pathway_file.close()
                
        return gene_dict

    def expand_groups(self, node_id, groups):
        """
        node_id: a node ID that may be a group
        groups: store group IDs and list of sub-ids
        return value: a list that contains all group IDs deconvoluted
        """
        node_list = []
        if node_id in groups.keys():
            for component_id in groups[node_id]:
                node_list.extend(self.expand_groups(component_id, groups))
        else:
            node_list.extend([node_id])
        return node_list
    
    def read_kegg(self, lines, kegg_dict, org_dict, load_subpaths: bool = False):
        """
        Parse a KEGG KGML file into a NetworkX directed graph (gene-gene complete graph).
        """
        soup = BeautifulSoup("".join(lines), "xml")

        groups = {}
        id_to_name = {}
        subpaths = []
        G = nx.DiGraph()

        # -----------------------------
        # Parse <entry> elements
        # -----------------------------
        for entry in soup.find_all("entry"):
            entry_id = entry.get("id")
            entry_type = entry.get("type", "")
            entry_name_raw = entry.get("name", "")
            parts = entry_name_raw.split(":")

            if len(parts) == 1:
                canonical_name = parts[0]
            elif parts[0] in {self.organism, "ko"}:
                d = org_dict if parts[0] == self.organism else kegg_dict
                genes = [p.split()[0] for p in parts[1:]]
                names = [d.get(g, g) for g in genes]
                canonical_name = "-".join(names)
            elif parts[0] == "path":
                canonical_name = parts[-1]
                entry_type = "path"
            else:
                canonical_name = entry_name_raw

            canonical_name = re.sub(",", "", canonical_name)
            id_to_name[entry_id] = canonical_name
            if entry_type == "path":
                subpaths.append(canonical_name)

            # store group members
            if entry_type == "group":
                groups[entry_id] = [c["id"] for c in entry.find_all("component")]
            else:
                G.add_node(canonical_name, name=canonical_name, type=entry_type)
                if entry_type in {"gene", "enzyme"}:
                    G.nodes[canonical_name]["type"] = "gene"

        # -----------------------------
        # Parse <relation> (signaling)
        # -----------------------------
        for rel in soup.find_all("relation"):
            e1, e2 = rel.get("entry1"), rel.get("entry2")
            subtypes = [s.get("name", "") for s in rel.find_all("subtype")]
            relation_type = rel.get("type", "")

            signal = None
            if any(s in subtypes for s in ["activation", "expression", "glycosylation"]):
                color, signal = "green", "a"
            elif any(s in subtypes for s in ["inhibition", "repression"]):
                color, signal = "red", "i"
            elif "dephosphorylation" in subtypes:
                color, signal = "pink", "i"
            elif "ubiquitination" in subtypes:
                color, signal = "cyan", "i"
            elif "phosphorylation" in subtypes:
                color, signal = "orange", "a"
            else:
                color, signal = "black", None

            for a, b in itertools.product(
                self.expand_groups(e1, groups), self.expand_groups(e2, groups)
            ):
                if a in id_to_name and b in id_to_name:
                    G.add_edge(
                        id_to_name[a],
                        id_to_name[b],
                        color=color,
                        subtype="/".join(subtypes),
                        type=relation_type,
                        signal=signal or "0",
                    )

        # -----------------------------
        # Parse <reaction> (metabolic)
        # -----------------------------
        # Map reaction IDs to genes
        reaction_to_genes = defaultdict(set)
        for entry in soup.find_all("entry"):
            if entry.get("type") in {"gene", "enzyme"} and entry.get("reaction"):
                for rid in entry.get("reaction").split():
                    reaction_to_genes[rid.replace("rn:", "")].add(id_to_name[entry["id"]])

        # Parse reaction substrates/products
        for rxn in soup.find_all("reaction"):
            rid = rxn.get("name", "").replace("rn:", "")
            subs = [s["name"] for s in rxn.find_all("substrate")]
            prods = [p["name"] for p in rxn.find_all("product")]

            # add compound nodes
            for c in subs + prods:
                if not G.has_node(c):
                    G.add_node(c, type="compound")

            # connect gene → reaction compounds
            for g in reaction_to_genes.get(rid, []):
                for p in prods:
                    G.add_edge(g, p, type="metabolic_product", signal="0")
                for s in subs:
                    G.add_edge(s, g, type="metabolic_substrate", signal="0")

        # -----------------------------
        # Collapse intermediates: gene→compound→gene
        # -----------------------------
        collapsed = nx.DiGraph()
        for gene in [n for n, d in G.nodes(data=True) if d.get("type") == "gene"]:
            for compound in G.successors(gene):
                if G.nodes[compound].get("type") == "compound":
                    downstream = [
                        n for n in G.successors(compound)
                        if G.nodes[n].get("type") == "gene"
                    ]
                    for g2 in downstream:
                        if gene != g2:
                            collapsed.add_edge(gene, g2, type="metabolic_influence", signal="0")

        G_complete = nx.compose(G, collapsed)
                
        return G_complete

    def write_xml_files(self, organism: str, *, max_workers: int = 4, min_interval_s: float = 0.8):
        logging.info("\t\tDownloading pathway files in parallel...")

        base_dir = Path(self.file_paths["pathway_xml_files"]) / organism
        base_dir.mkdir(parents=True, exist_ok=True)

        existing_xml = {Path(f).stem for f in os.listdir(base_dir) if f.endswith(".xml")}

        def _norm_num(p: str) -> str | None:
            raw = p.replace("path:", "")
            num = re.sub(r"[a-zA-Z]+", "", raw)
            return num or None

        nums = [_norm_num(p) for p in self.pathway_list]
        nums = [n for n in nums if n is not None]

        tasks = []
        for n in nums:
            code = f"{organism}{n}"
            if code not in existing_xml:
                tasks.append((code, base_dir / f"{code}.xml"))

        if not tasks:
            logging.info("\t\tAll KGML files already present.")
            return

        # ----- ONE shared Session + throttle + adaptive slow-down -----
        retry = Retry(
            total=4, connect=4, read=4,
            backoff_factor=0.25,                     # 0.25, 0.5, 1.0, 2.0 ...
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "CustomGRN-KGMLFetcher/1.0 (+https://example.org)",
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
        })
        # pool_maxsize ~= max_workers helps reuse connections
        adapter = HTTPAdapter(max_retries=retry, pool_connections=max_workers, pool_maxsize=max_workers)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)

        throttle_lock = threading.Lock()
        last_t = {"t": 0.0}
        base_interval = float(min_interval_s)
        dyn_interval = {"v": base_interval}

        # simple sliding-window failure counter to slow down if needed
        fail_lock = threading.Lock()
        recent_fails = []

        def _throttled_get(url, timeout=(8, 30)):
            with throttle_lock:
                now = time.monotonic()
                wait = last_t["t"] + dyn_interval["v"] - now
                if wait > 0:
                    time.sleep(wait)
                r = sess.get(url, stream=True, timeout=timeout)
                last_t["t"] = time.monotonic()
            return r

        def _note_fail():
            # keep the last ~30 seconds of failures
            with fail_lock:
                now = time.monotonic()
                recent_fails[:] = [t for t in recent_fails if now - t < 30.0]
                recent_fails.append(now)
                # if > 8 failures in last 30s, increase pace modestly
                if len(recent_fails) > 8:
                    dyn_interval["v"] = min(2.0, dyn_interval["v"] * 1.3)  # cap at 2s
                else:
                    # slowly relax back toward base
                    dyn_interval["v"] = max(base_interval, dyn_interval["v"] * 0.95)

        def _note_success():
            with fail_lock:
                # gently move back toward base interval
                dyn_interval["v"] = max(base_interval, dyn_interval["v"] * 0.9)

        def _fetch_one(code: str, out_path: Path):
            if out_path.exists() and out_path.stat().st_size > 0:
                return (code, True, "exists")

            urls_primary = f"https://www.kegg.jp/kegg-bin/download?entry={code}&format=kgml"
            urls_fallback = f"https://rest.kegg.jp/get/{code}/kgml"

            for attempt in range(2):  # at most two endpoint families: CGI then REST
                url = urls_primary if attempt == 0 else urls_fallback
                try:
                    r = _throttled_get(url, timeout=(8, 30))
                    if r.status_code != 200:
                        _note_fail()
                        continue
                    text = r.text
                    if "<pathway" not in text:
                        _note_fail()
                        continue
                    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(text)
                    tmp.replace(out_path)
                    _note_success()
                    return (code, True, "ok")
                except (RemoteDisconnected, requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                    # immediate fallback to the other endpoint on transport issues
                    _note_fail()
                    continue
                except Exception as e:
                    _note_fail()
                    # small jitter before retrying anything else
                    time.sleep(base_interval + random.random() * 0.2)
                    continue

            return (code, False, "failed")

        # ----- run pool -----
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_one, code, outp): (code, outp) for code, outp in tasks}
            ok = 0; errs = []
            for fut in tqdm(as_completed(futs), total=len(futs), desc="KGML", unit="file", dynamic_ncols=True):
                code, outp = futs[fut]
                try:
                    _, success, note = fut.result()
                    if success: ok += 1
                    else: errs.append((code, note))
                except Exception as e:
                    errs.append((code, f"Worker exception: {e!r}"))

        if errs:
            for c, e in errs[:8]:
                logging.debug(f"\t\tFailed {c}: {e}")
            logging.warning(f"\t\tCompleted {ok}/{len(tasks)} KGML files; {len(errs)} failed. "
                            f"(throttle now {dyn_interval['v']:.2f}s)")
        else:
            logging.info(f"\t\tCompleted {ok}/{len(tasks)} KGML files "
                        f"(throttle {dyn_interval['v']:.2f}s).")



    def parse_kegg_pathway(self, graph, pathway_code, pathway_num, num_pathways):
        """
        Clean and format a KEGG pathway graph:
        - Split complex nodes (A-B) into separate genes
        - Rewire incoming/outgoing edges
        - Remove redundant group edges and self-loops
        """
        # --- Attach pathway metadata ---
        graph.graph["pathway"] = pathway_code
        graph.graph["source"] = "ko" if pathway_code.startswith("ko") else self.organism

        # --- Rewire complexes ---
        remove_nodes = [n for n in graph.nodes if "-" in n]
        for rm in remove_nodes:
            preds = list(graph.predecessors(rm))
            succs = list(graph.successors(rm))
            parts = rm.split("-")

            for start in preds:
                edge_data = graph.get_edge_data(start, rm) or {}
                signal = edge_data.get("signal", "a")
                for g in parts:
                    graph.add_edge(start, g, **edge_data)

            for end in succs:
                edge_data = graph.get_edge_data(rm, end) or {}
                signal = edge_data.get("signal", "a")
                for g in parts:
                    graph.add_edge(g, end, **edge_data)

            graph.remove_node(rm)

        # --- Remove redundant group edges ---
        for node in list(graph.nodes()):
            preds = list(graph.predecessors(node))
            for pred in preds:
                if "-" in pred:
                    genes = pred.split("-")
                    if all(g in preds for g in genes) and graph.has_edge(pred, node):
                        graph.remove_edge(pred, node)

        # --- Remove self-loops ---
        graph.remove_edges_from([(u, v) for u, v in graph.edges if u == v])

        logging.info(f'Pathway ({pathway_num}/{num_pathways}): {pathway_code} Edges: {len(graph.edges())}')

        # Optional metadata
        if pathway_code.startswith(self.organism):
            graph.graph["source"] = self.organism
        elif pathway_code.startswith("ko"):
            graph.graph["source"] = "ko"

        # Save graph
        out_file = os.path.join(self.output_path, f"{pathway_code}.graphml")
        
        graph = self._sanitize_for_graphml(graph)
        nx.write_graphml(graph, out_file)
        self.pathway_dict[pathway_code] = graph


    def find_kegg_pathways(self):
        """
        write_graphml = whether or not to write out a graphml (usually true)
        organism = organism code from kegg. Eg human = 'hsa', mouse = 'mus'

        Finds the KEGG pathways from the pathway dictionaries
        """
        organism = self.organism

        logging.info("\t\tFinding KEGG pathways...")
        kegg_dict = self.parse_kegg_dict()  # parse the dictionary of ko codes
        logging.info("\t\t\tLoaded KEGG code dictionary")
        
        pathway_dict_path = f'{self.file_paths["pathway_xml_files"]}/{organism}_dict.csv'
        alias_dict = {}
        org_dict = {}

        # If the dictionary file exists, use that (much faster than streaming)
        if f'{organism}_dict.csv' in os.listdir(f'{self.file_paths["pathway_xml_files"]}'):
            logging.info(f'\t\t\tReading {organism} dictionary file...')
            with open(pathway_dict_path, 'r') as kegg_dict_file:
                for line in kegg_dict_file:
                    line = line.strip().split('\t')
                    k = line[0]
                    name = line[1]
                    org_dict[k] = name

        # If the dictionary file does not exist, write it and stream in the data for the dictionary
        else:
            logging.info(f'\t\t\tOrganism dictionary not present for {organism}, downloading...')
            try:  # try to retrieve and parse the dictionary containing organism gene names to codes conversion
                url = requests.get(f"https://rest.kegg.jp/list/{organism}", stream=True, timeout=60)
                # reads KEGG dictionary of identifiers between numbers and actual protein names and saves it to a python dictionary

                with open(pathway_dict_path, 'w') as kegg_dict_file:
                    for line in url.iter_lines():
                        line = line.decode("utf-8")
                        line_split = line.split("\t")
                        k = line_split[0].split(":")[1]
                        nameline = line_split[3].split(";")
                        name = nameline[0]
                        if "," in name:
                            nameline = name.split(",")
                            name = nameline[0]
                            for entry in range(1, len(nameline)):
                                alias_dict[nameline[entry].strip()] = name.upper()
                        org_dict[k] = name
                        kegg_dict_file.write(f'{k}\t{name}\n')
                url.close()
            except:
                logging.info("Could not get library: " + organism)
        
        # Writes xml files for the pathways in the pathway list
        self.write_xml_files(organism)

        xml_dir = Path(self.file_paths["pathway_xml_files"]) / organism
        xml_file_names = set(os.listdir(xml_dir))

        def parse_xml_files(xml_file):
            """
            Reads in the pathway xml file and parses the connections. Creates a networkx directed graph of the pathway
            """
            with open(f'{self.file_paths["pathway_xml_files"]}/{organism}/{xml_file}', 'r') as pathway_file:
                text = [line for line in pathway_file]

                # Read the kegg xml file
                graph = self.read_kegg(text, kegg_dict, org_dict)

                # Parse the kegg pathway
                pathway_code = xml_file.replace(".xml", "")
                self.parse_kegg_pathway(graph, pathway_code, pathway_num, num_pathways)

        num_pathways = len(self.pathway_list)

        for pathway_num, pathway in enumerate(self.pathway_list):
            num = re.sub(r"[a-zA-Z]+", "", pathway)
            
            org_file = f"{organism}{num}.xml"
            
            if org_file in xml_file_names:
                parse_xml_files(org_file)
            else:
                logging.info(f"No KGML file found for pathway {num}")
                            
    def _sanitize_for_graphml(self, G: nx.DiGraph):
        for n, attrs in G.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, type):  # e.g. <class 'str'>
                    G.nodes[n][k] = str(v)
                elif v is None:
                    G.nodes[n][k] = "0"
        for u, v, attrs in G.edges(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, type):
                    G.edges[u, v][k] = str(v)
                elif v is None:
                    G.edges[u, v][k] = "0"
        return G

    def add_pathways(self, pathway_list, write_graphml=True, removeSelfEdges=False, organism='hsa'):
        """
        Add a list of pathways in graphml format to the rule_inference object

        Writes out the "_processed.graphml" files
        """

        logging.info(f'\t\tAdding graphml pathways to rule_inference object...')

        def create_processed_networkx_graphml(G, pathway):
            """
            Reads in the graph and the pathway and filters out self edges and isolates

            Creates the "_processed.graphml" files
            """
            nodes = set(G.nodes())

            logging.info(f'\t\tPathway: {pathway} Edges: {len(G.edges())}')
            nodes = list(G.nodes())

            if removeSelfEdges:
                G.remove_edges_from(nx.selfloop_edges(G))  # remove self loops

            # graph post-processing
            # remove singletons/isolates
            G.remove_nodes_from(list(nx.isolates(G)))

            self.pathway_graphs[pathway] = G
            logging.info(f'\t\t\tEdges after processing: {len(G.edges())}')

            if write_graphml:
                base = self.output_path
                fname = pathway
                if not fname.startswith(organism):
                    fname = f"{organism}{fname}"
                if not fname.endswith("_processed.graphml"):
                    fname = f"{fname}_processed.graphml"
                out_path = os.path.join(base, fname)
                
                
                graph = self._sanitize_for_graphml(G)
                nx.write_graphml(graph, out_path, infer_numeric_types=True)

        # If pathway_list is a list
        if isinstance(pathway_list, list):
            for pathway in pathway_list:  
                if os.path.exists(pathway):
                    G = nx.read_graphml(pathway)
                create_processed_networkx_graphml(G, pathway)
                

        # If pathway_list is a dictionary
        elif isinstance(pathway_list, dict):
                for pathway, G in pathway_list.items():
                    create_processed_networkx_graphml(G, pathway)
        
    def build_global_network(self, write_graphml=True, filename="all_kegg_pathways.graphml"):
        """
        Combine every pathway graph currently in `self.pathway_dict`
        into a single directed graph and (optionally) write it to disk.
        """
        if not self.pathway_dict:
            raise ValueError("`pathway_dict` is empty – run `find_kegg_pathways` first")

        # Union of all edges/nodes (attributes are preserved; duplicates collapsed)
        G_global = nx.compose_all(self.pathway_dict.values())

        # Tag every edge with the pathway(s) it came from
        for path_code, g in self.pathway_dict.items():
            for u, v, data in g.edges(data=True):
                if (u, v) in G_global.edges:
                    G_global[u][v].setdefault("pathways", set()).add(path_code)

        pathway_like_nodes = [
            n for n in G_global.nodes
            if re.match(r"^(PATH:|MMU\d{5}|HSA\d{5}|KO\d{5})$", str(n).upper())
        ]
        
        # Remove pathway-like nodes
        G_global.remove_nodes_from(pathway_like_nodes)
        
        print(f"Total nodes: {G_global.number_of_nodes():,}")
        print(f"Pathway-like nodes remaining: {len(pathway_like_nodes):,}")
        print("Examples:", pathway_like_nodes[:10])

        # Write the graph to disk (after converting unsupported types)
        if write_graphml:
            for u, v, d in G_global.edges(data=True):
                if isinstance(d.get("pathways"), set):
                    d["pathways"] = ",".join(sorted(d["pathways"]))

            out_path = os.path.join(self.output_path, filename)
            graph = self._sanitize_for_graphml(G_global)
            nx.write_graphml(graph, out_path, infer_numeric_types=True)
            logging.info(f"Wrote merged network with {G_global.number_of_nodes()} nodes "
                        f"and {G_global.number_of_edges()} edges to {out_path}")

        return G_global

def collapse_metabolites(G: nx.DiGraph) -> nx.DiGraph:
    """
    Collapse compound nodes (CPD:Cxxxxx) into direct gene–gene edges.

    Converts gene → compound → gene paths into gene → gene edges
    with type='metabolic_influence' and signal='0'.
    Propagates 'pathways' provenance from adjacent edges.
    Removes compound nodes entirely.
    """
    import re
    collapsed = nx.DiGraph()

    # --- Identify KEGG compound nodes ---
    compound_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "compound" or re.match(r"^CPD:C\d{5}$", str(n))
    ]

    for c in compound_nodes:
        # Find all upstream and downstream gene nodes
        upstream_genes = [
            u for u in G.predecessors(c)
            if G.nodes[u].get("type") == "gene"
        ]
        downstream_genes = [
            v for v in G.successors(c)
            if G.nodes[v].get("type") == "gene"
        ]

        # Create gene–gene influence edges
        for g1 in upstream_genes:
            for g2 in downstream_genes:
                if g1 == g2:
                    continue

                # Collect pathways from both source→compound and compound→target edges
                pathways = set()
                if G.has_edge(g1, c):
                    p1 = G[g1][c].get("pathways")
                    if isinstance(p1, str):
                        pathways.update(p1.split(","))
                    elif isinstance(p1, (list, set)):
                        pathways.update(p1)
                if G.has_edge(c, g2):
                    p2 = G[c][g2].get("pathways")
                    if isinstance(p2, str):
                        pathways.update(p2.split(","))
                    elif isinstance(p2, (list, set)):
                        pathways.update(p2)

                collapsed.add_edge(
                    g1,
                    g2,
                    type="metabolic_influence",
                    signal="0",
                    via=c,
                    pathways=",".join(sorted(p for p in pathways if p)),
                )

    # --- Remove compound nodes and merge collapsed edges ---
    G_no_cpds = G.copy()
    G_no_cpds.remove_nodes_from(compound_nodes)
    G_final = nx.compose(G_no_cpds, collapsed)

    return G_final


def build_kegg_pkn(
    dataset_name: str,
    output_path: Union[Path, str],
    out_csv: Union[str, None],
    out_gpickle: Union[str, None],
    organism: str = "mmu",          # KEGG org code (e.g., 'mmu', 'hsa'); 'mm10' gets normalized in your code if you added that
    normalize_case: str = "upper",  # "upper" | "lower" | None
):
    """
    Build the FULL KEGG PKN by parsing all KGML pathways into a global directed graph,
    then flattening to an edge list with sign and provenance.

    Output columns:
      TF, TG, kegg_signal (-1/0/+1), kegg_n_pathways, kegg_pathways
    """
    # 1) Build (or reuse cached) KEGG global network
    pw = Pathways(
        dataset_name=dataset_name,
        output_path=output_path,
        organism=organism
    )

    logging.info("Discovering KEGG pathways…")
    pw.find_kegg_pathways()

    logging.info("Composing global KEGG network…")
    G = pw.build_global_network(filename="all_kegg_pathways.graphml")
    
    G = collapse_metabolites(G)
    
    G = pw._sanitize_for_graphml(G)

    # 2) Flatten graph to PKN
    rows = []
    sign_map = {"a": 1, "i": -1}

    for u, v, d in G.edges(data=True):
        signal = d.get("signal")  # 'a' (activation) / 'i' (inhibition) / None
        sign = sign_map.get(signal, 0)

        paths = d.get("pathways", "")
        if isinstance(paths, set):
            paths = ",".join(sorted(paths))
        n_paths = 0 if not paths else len([p for p in paths.split(",") if p])

        rows.append({
            "TF": u,
            "TG": v,
            "kegg_signal": sign,
            "kegg_n_pathways": n_paths,
            "kegg_pathways": paths
        })

    pkn = pd.DataFrame(rows).drop_duplicates()

    def _canon(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        if normalize_case == "upper":
            return s.str.upper()
        if normalize_case == "lower":
            return s.str.lower()
        return s

    if not pkn.empty:
        pkn["TF"] = _canon(pkn["TF"])
        pkn["TG"] = _canon(pkn["TG"])

    # optional outputs
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        pkn.to_csv(out_csv, index=False)
        logging.info(f"Wrote KEGG PKN CSV → {out_csv}")

    if out_gpickle:
        with open(out_gpickle, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Wrote KEGG PKN GraphML → {out_gpickle}")
