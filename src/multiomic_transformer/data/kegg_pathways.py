from bioservices.kegg import KEGG
import networkx as nx
import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import itertools
# from rule_inference import *
import logging
import os
from alive_progress import alive_bar
import sys
import logging
from config.settings import *

# from self.file_paths import self.file_paths

class Pathways:
    """
    Reads in and processes the KEGG pathways as networkx graphs
    """
    def __init__(self, dataset_name, cv_threshold, output_path, data_file, gene_name_file, write_graphml, organism):
        self.cv_threshold = cv_threshold # The cutoff value threshold for binarizing 
        self.data_file = data_file
        self.gene_list = self._find_genes(gene_name_file)
        self.pathway_graphs = {}
        self.dataset_name = dataset_name
        self.output_path = os.path.join(output_path, "graphml_files/")
        self.gene_indices = []
        self.pathway_dict = {}
        self.organism = organism
        self.file_paths = {"pathway_xml_files": output_path}
        
        if self.cv_threshold:
            self.filter_data()

        os.makedirs(self.output_path, exist_ok=True)

    def _count_generator(self, reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)
    
    def _find_genes(self, gene_name_file):
        """
        Finds the names of the genes in the datafile
        """
        
        gene_names_df = pd.read_csv(gene_name_file, header=None, index_col=0)
        gene_list = gene_names_df.index.to_list()
        
        return gene_list
    
    def filter_data(self):
        """
        Filters out genes with low variability. The threshold is low by default (0.001) to allow for most genes
        to be kept in. Increasing this value will lead to only including highly variable genes
        """
        logging.info(f'\tFiltering data based on cv threshold of {self.cv_threshold}')
        self.cv_genes = []
        with open(self.data_file, "r") as file:
            next(file)
            for line in file:
                column = line.split(',')
                gene_name = column[0]
                row_data = [float(cell_value) for cell_value in column[1:]]
                
                # Calculate the cutoff value 
                if np.std(row_data) / np.mean(row_data) >= self.cv_threshold:
                    if gene_name in self.gene_list:
                        self.cv_genes.append(gene_name)

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

            pathway_file = requests.get("http://rest.kegg.jp/get/br:ko00001", stream=True)
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
    
    def read_kegg(self, lines, graph, KEGGdict, hsaDict):
        # read all lines into a bs4 object using libXML parser
        logging.info('\t\tReading KEGG xml file')
        soup = BeautifulSoup("".join(lines), "xml")
        groups = {}  # store group IDs and list of sub-ids
        id_to_name = {}  # map id numbers to names
        subpaths = []
        
        # Look at each entry in the kgml file. Info: (https://www.kegg.jp/kegg/xml/)
        for entry in soup.find_all("entry"):

            # Name of each gene in the entry
            # If there are multiple genes in the entry, store them all with the same id
            entry_split = entry["name"].split(":")

            # If the entry is part of a group (in the network coded by a group containing lots of related genes)
            if len(entry_split) > 2:

                # Choose which dictionary to use based on whether the entries are hsa or kegg elements
                # Entries with hsa correspond to genes, entries with ko correspond to orthologs
                if entry_split[0] == self.organism or entry_split[0] == "ko":
                    if entry_split[0] == self.organism:
                        useDict = hsaDict
                    elif entry_split[0] == "ko":
                        useDict = KEGGdict
                    nameList = []
                    
                    # Split off the first name
                    entry_name = ""
                    namer = entry_split.pop(0)
                    namer = entry_split.pop(0)
                    namer = namer.split()[0]

                    # Either use the dictionary name for the key or use the name directly if its not in the dictionary
                    entry_name = (
                        entry_name + useDict[namer]
                        if namer in useDict.keys()
                        else entry_name + namer
                    )

                    # Append each gene name to the list ([gene1, gene2])
                    for i in range(len(entry_split)):
                        nameList.append(entry_split[i].split()[0])

                    # For each of the gene names, separate them with a "-" (gene1-gene2)
                    for namer in nameList:
                        entry_name = (
                            entry_name + "-" + useDict[namer]
                            if namer in useDict.keys()
                            else entry_name + "-" + namer
                        )
                    entry_type = entry["type"]
                else:
                    entry_name = entry["name"]
                    entry_type = entry["type"]
            
            # If there is only one name
            else:
                # If the name is hsa
                if entry_split[0] == self.organism:
                    entry_name = entry_split[1] # Get the entry number
                    entry_type = entry["type"] # Get the entry type
                    entry_name = ( # Get the gene name from the entry number if its in the hsa gene name dict
                        hsaDict[entry_name] if entry_name in hsaDict.keys() else entry_name
                    )
                # If the name is ko, do the same as above but use the KEGGdict instead of the hsa gene name dict
                elif entry_split[0] == "ko":
                    entry_name = entry_split[1]
                    entry_type = entry["type"]
                    entry_name = (
                        KEGGdict[entry_name]
                        if entry_name in KEGGdict.keys()
                        else entry_name
                    )
                # If the entry is another KEGG pathway number, store the name of the signaling pathway
                elif entry_split[0] == "path":
                    entry_name = entry_split[1]
                    entry_type = "path"
                # If none of the above, just store the name and type
                else:
                    entry_name = entry["name"]
                    entry_type = entry["type"]
            
            # Get the unique entry ID for this pathway
            entry_id = entry["id"]

            # Some genes will have ',' at the end if there were more than one gene, remove that
            entry_name = re.sub(",", "", entry_name)

            # Map the id of the entry to the entry name
            id_to_name[entry_id] = entry_name
            # logging.info(f'Entry name: {entry_name} ID: {entry_id}')

            # If the entry is a pathway, store the pathway name
            if entry_type == "path":
                if entry_name not in subpaths:
                    subpaths.append(entry_name)
                

            # If the entry type is a gene group, find all component ids and add them to the id dictionary for the entry
            if entry_type == "group":
                group_ids = []
                for component in entry.find_all("component"):
                    group_ids.append(component["id"])
                groups[entry_id] = group_ids
            
            # If the entry is not a group, add its attributes to the graph
            else:
                graph.add_node(entry_name, name=entry_name, type=entry_type)

        # For each of the relationships
        for relation in soup.find_all("relation"):
            (color, signal) = ("black", "a")

            relation_entry1 = relation["entry1"] # Upstream node
            relation_entry2 = relation["entry2"] # Target node
            relation_type = relation["type"] # Type of relationship (PPel, GEcrel, etc.)
    
            subtypes = []

            # Relationship subtypes tell you about whats going on
            for subtype in relation.find_all("subtype"):
                subtypes.append(subtype["name"])
    
            if (
                ("activation" in subtypes)
                or ("expression" in subtypes)
                or ("glycosylation" in subtypes)
            ):
                color = "green"
                signal = "a"
            elif ("inhibition" in subtypes) or ("repression" in subtypes):
                color = "red"
                signal = "i"
            elif ("binding/association" in subtypes) or ("compound" in subtypes):
                color = "purple"
                signal = "a"
            elif "phosphorylation" in subtypes:
                color = "orange"
                signal = "a"
            elif "dephosphorylation" in subtypes:
                color = "pink"
                signal = "i"
            elif "indirect effect" in subtypes:
                color = "cyan"
                signal = "a"
            elif "dissociation" in subtypes:
                color = "yellow"
                signal = "i"
            elif "ubiquitination" in subtypes:
                color = "cyan"
                signal = "i"
            else:
                logging.debug("color not detected. Signal assigned to activation arbitrarily")
                logging.debug(subtypes)
                signal = "a"

            # For entries that are a group of genes, get a list of all of the sub-id's in that group
            entry1_list = self.expand_groups(relation_entry1, groups)
            entry2_list = self.expand_groups(relation_entry2, groups)

            # Find all connections between objects in the groups and add them to the grapgh
            for (entry1, entry2) in itertools.product(entry1_list, entry2_list):
                node1 = id_to_name[entry1]
                node2 = id_to_name[entry2]
                # if (node1.count('-') < 10 or node2.count('-') < 10):
                #     logging.info(f'{node1} --- {signal} ---> {node2}\n\t{"/".join(subtypes)}')
                graph.add_edge(
                    node1,
                    node2,
                    color=color,
                    subtype="/".join(subtypes),
                    type=relation_type,
                    signal=signal,
                )
        
        # ------------------ UNCOMMENTING THE FOLLOWING LOADS ALL REFERENCED SUBGRAPHS, WORK IN PROGRESS -------------------------------
        logging.info(f'Subpaths:')
        for path_name in subpaths:
            logging.info(f'\t{path_name}')

        num_pathways = len(subpaths)
        for pathway_num, pathway in enumerate(subpaths):
            for xml_file in os.listdir(f'{self.file_paths["pathway_xml_files"]}/{self.organism}'):
                xml_pathway_name = xml_file.split('.')[0]
                if pathway == xml_pathway_name:
                    with open(f'{self.file_paths["pathway_xml_files"]}/{self.organism}/{xml_file}', 'r') as pathway_file:
                        text = [line for line in pathway_file]
                    soup = BeautifulSoup("".join(text), "xml")
                    for entry in soup.find_all("entry"):
                        # logging.info(f'\nEntry:')
                        # logging.info(f'\t{entry}')

                        # Name of each gene in the entry
                        # If there are multiple genes in the entry, store them all with the same id
                        entry_split = entry["name"].split(":")

                        # logging.info(f'\tentry_split: {entry_split}')
                        # logging.info(f'\tlen(entry_split) : {len(entry_split)}')
                        # If the entry is part of a group (in the network coded by a group containing lots of related genes)
                        if len(entry_split) > 2:

                            # Choose which dictionary to use based on whether the entries are hsa or kegg elements
                            # Entries with hsa correspond to genes, entries with ko correspond to orthologs
                            if entry_split[0] == self.organism or entry_split[0] == "ko":
                                if entry_split[0] == self.organism:
                                    useDict = hsaDict
                                elif entry_split[0] == "ko":
                                    useDict = KEGGdict
                                nameList = []
                                
                                # Split off the first name
                                entry_name = ""
                                namer = entry_split.pop(0)
                                namer = entry_split.pop(0)
                                namer = namer.split()[0]

                                # Either use the dictionary name for the key or use the name directly if its not in the dictionary
                                entry_name = (
                                    entry_name + useDict[namer]
                                    if namer in useDict.keys()
                                    else entry_name + namer
                                )

                                # Append each gene name to the list ([gene1, gene2])
                                for i in range(len(entry_split)):
                                    nameList.append(entry_split[i].split()[0])

                                # For each of the gene names, separate them with a "-" (gene1-gene2)
                                for namer in nameList:
                                    entry_name = (
                                        entry_name + "-" + useDict[namer]
                                        if namer in useDict.keys()
                                        else entry_name + "-" + namer
                                    )
                                entry_type = entry["type"]
                            else:
                                entry_name = entry["name"]
                                entry_type = entry["type"]
                        
                        # If there is only one name
                        else:
                            # If the name is hsa
                            if entry_split[0] == self.organism:
                                entry_name = entry_split[1] # Get the entry number
                                entry_type = entry["type"] # Get the entry type
                                entry_name = ( # Get the gene name from the entry number if its in the hsa gene name dict
                                    hsaDict[entry_name] if entry_name in hsaDict.keys() else entry_name
                                )
                            # If the name is ko, do the same as above but use the KEGGdict instead of the hsa gene name dict
                            elif entry_split[0] == "ko":
                                entry_name = entry_split[1]
                                entry_type = entry["type"]
                                entry_name = (
                                    KEGGdict[entry_name]
                                    if entry_name in KEGGdict.keys()
                                    else entry_name
                                )
                            # If the entry is another KEGG pathway number, store the name of the signaling pathway
                            elif entry_split[0] == "path":
                                entry_name = entry_split[1]
                                entry_type = "path"
                            # If none of the above, just store the name and type
                            else:
                                entry_name = entry["name"]
                                entry_type = entry["type"]
                        
                        # Get the unique entry ID for this pathway
                        entry_id = entry["id"]

                        # Some genes will have ',' at the end if there were more than one gene, remove that
                        entry_name = re.sub(",", "", entry_name)

                        # Map the id of the entry to the entry name
                        id_to_name[entry_id] = entry_name
                        # logging.info(f'Entry name: {entry_name} ID: {entry_id}')

                        # # If the entry is a pathway, store the pathway name
                        # if entry_type == "path":
                        #     if entry_name not in subpaths:
                        #         subpaths.append(entry_name)
                            

                        # If the entry type is a gene group, find all component ids and add them to the id dictionary for the entry
                        if entry_type == "group":
                            group_ids = []
                            for component in entry.find_all("component"):
                                group_ids.append(component["id"])
                            groups[entry_id] = group_ids
                        
                        # If the entry is not a group, add its attributes to the graph
                        else:
                            graph.add_node(entry_name, name=entry_name, type=entry_type)

                    # For each of the relationships
                    for relation in soup.find_all("relation"):
                        # logging.info(f'Relation:')
                        # logging.info(f'\t{relation}')
                        (color, signal) = ("black", "a")

                        relation_entry1 = relation["entry1"] # Upstream node
                        relation_entry2 = relation["entry2"] # Target node
                        relation_type = relation["type"] # Type of relationship (PPel, GEcrel, etc.)
                
                        subtypes = []

                        # Relationship subtypes tell you about whats going on
                        for subtype in relation.find_all("subtype"):
                            subtypes.append(subtype["name"])
                
                        if (
                            ("activation" in subtypes)
                            or ("expression" in subtypes)
                            or ("glycosylation" in subtypes)
                        ):
                            color = "green"
                            signal = "a"
                        elif ("inhibition" in subtypes) or ("repression" in subtypes):
                            color = "red"
                            signal = "i"
                        elif ("binding/association" in subtypes) or ("compound" in subtypes):
                            color = "purple"
                            signal = "a"
                        elif "phosphorylation" in subtypes:
                            color = "orange"
                            signal = "a"
                        elif "dephosphorylation" in subtypes:
                            color = "pink"
                            signal = "i"
                        elif "indirect effect" in subtypes:
                            color = "cyan"
                            signal = "a"
                        elif "dissociation" in subtypes:
                            color = "yellow"
                            signal = "i"
                        elif "ubiquitination" in subtypes:
                            color = "cyan"
                            signal = "i"
                        else:
                            logging.debug("color not detected. Signal assigned to activation arbitrarily")
                            logging.debug(subtypes)
                            signal = "a"

                        # For entries that are a group of genes, get a list of all of the sub-id's in that group
                        entry1_list = self.expand_groups(relation_entry1, groups)
                        entry2_list = self.expand_groups(relation_entry2, groups)

                        # Find all connections between objects in the groups and add them to the grapgh
                        for (entry1, entry2) in itertools.product(entry1_list, entry2_list):
                            node1 = id_to_name[entry1]
                            node2 = id_to_name[entry2]
                            # if (node1.count('-') < 10 or node2.count('-') < 10):
                            # logging.info(f'{node1} --- {signal} ---> {node2}\n\t{"/".join(subtypes)}')
                            graph.add_edge(
                                node1,
                                node2,
                                color=color,
                                subtype="/".join(subtypes),
                                type=relation_type,
                                signal=signal,
                            )


        ### --------------------------------------------------------------------------------------------------


        # self.add_pathways(subpath_graphs, minOverlap=25, organism=self.organism)

        return graph
    

    def write_xml_files(self, organism: str, pathway_list: list):
        """
        Reads in all xml files for the organism, faster to do this once at the start and just use
        the cached files. They aren't that big, so I'd rather store them at the beginning.
        """
        # Function to silence KEGG initialization to the terminal
        def silent_kegg_initialization():
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
        k.organism = organism
        if not pathway_list:
            logging.info(f"No pathway list provided — downloading all KEGG pathways for {organism}...")
            try:
                response = requests.get(f"http://rest.kegg.jp/list/pathway/{organism}")
                response.raise_for_status()
                # KEGG returns lines like: "path:hsa00010\tGlycolysis / Gluconeogenesis"
                pathway_list = [line.split("\t")[0].replace("path:", "") for line in response.text.strip().split("\n")]
            except Exception as e:
                raise RuntimeError(f"Failed to fetch KEGG pathway list for {organism}: {e}")
        
        logging.info(f'\t\tDownloading pathway files, this may take a while...')
        base_dir = f'{self.file_paths["pathway_xml_files"]}/{organism}'
        os.makedirs(base_dir, exist_ok=True)

        def _fetch_kgml(code: str, out_path: str) -> None:
            """Fetch KEGG KGML for `code` (e.g., 'ko00010', 'mmu00010') to `out_path` if missing."""
            if os.path.exists(out_path):
                return
            try:
                resp = requests.get(f"http://rest.kegg.jp/get/{code}/kgml", stream=True, timeout=30)
                resp.raise_for_status()
                with open(out_path, "w") as f:
                    for chunk in resp.iter_lines():
                        if chunk:
                            f.write(chunk.decode("utf-8"))
            except Exception as e:
                logging.debug(f"could not read code: {code} ({e})")

        with alive_bar(len(pathway_list)) as bar:
            for pathway in pathway_list:
                # pathway may look like 'path:mmu04110' or 'mmu04110' or 'ko04110' — normalize to numeric id
                raw = pathway.replace("path:", "")
                num = re.sub(r"[a-zA-Z]+", "", raw)   # retain digits only
                if not num:
                    logging.debug(f"Skipping malformed pathway id: {pathway}")
                    bar()
                    continue

                ko_code  = f"ko{num}"
                org_code = f"{organism}{num}"

                ko_path  = os.path.join(base_dir, f"{ko_code}.xml")
                org_path = os.path.join(base_dir, f"{org_code}.xml")

                logging.debug(f'\t\t\tFetching {ko_code} and {org_code}')
                _fetch_kgml(ko_code, ko_path)
                _fetch_kgml(org_code, org_path)

                bar()

    def parse_kegg_pathway(self, graph, minimumOverlap, pathway_code, pathway_num, num_pathways):
        """
        Format and optionally write the KEGG pathway graph if it has sufficient overlap with gene list.

        Parameters
        ----------
        graph : networkx.DiGraph
            The parsed pathway graph.
        minimumOverlap : int
            Minimum number of overlapping genes required to keep the pathway.
        pathway_code : str
            Full KEGG code like 'ko00010' or 'hsa00010'.
        pathway_num : int
            Index of the current pathway being processed (for progress logging).
        num_pathways : int
            Total number of pathways (for progress logging).
        """
        # Remove complexes and rewire components
        removeNodeList = [gene for gene in graph.nodes() if "-" in gene]
        for rm in removeNodeList:
            for start in list(graph.predecessors(rm)):
                edge1 = graph.get_edge_data(start, rm)["signal"]
                for element in rm.split("-"):
                    graph.add_edge(start, element, signal=edge1)
            for finish in list(graph.successors(rm)):
                edge2 = graph.get_edge_data(rm, finish)["signal"]
                for element in rm.split("-"):
                    graph.add_edge(element, finish, signal=edge2)
            graph.remove_node(rm)

        # Remove redundant group dependency edges
        for node in list(graph.nodes()):
            predlist = list(graph.predecessors(node))
            for pred in predlist:
                if "-" in pred:
                    genes = pred.split("-")
                    if all(g in predlist for g in genes):
                        graph.remove_edge(pred, node)

        # Remove self-loops
        for u, v in list(graph.edges()):
            if u == v:
                graph.remove_edge(u, v)

        # Filter based on gene overlap
        pathway_nodes = set(graph.nodes())
        overlap = len(pathway_nodes.intersection(self.gene_list))

        if overlap > minimumOverlap and len(graph.edges()) > 0:
            logging.info(f'\t\t\tPathway ({pathway_num}/{num_pathways}): {pathway_code} '
                        f'Overlap: {overlap} Edges: {len(graph.edges())}')

            # Optional metadata
            if pathway_code.startswith(self.organism):
                graph.graph["source"] = self.organism
            elif pathway_code.startswith("ko"):
                graph.graph["source"] = "ko"

            # Save graph
            out_file = os.path.join(self.output_path, f"{pathway_code}.graphml")
            nx.write_graphml(graph, out_file)
            self.pathway_dict[pathway_code] = graph

        else:
            logging.debug(f'\t\t\tPathway ({pathway_num}/{num_pathways}): {pathway_code} '
                        f'not enough overlapping genes (min = {minimumOverlap}, found {overlap})')

    def find_kegg_pathways(self, kegg_pathway_list: list, write_graphml: bool, minimumOverlap: int):
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
        aliasDict = {}
        orgDict = {}

        # If the dictionary file exists, use that (much faster than streaming)
        if f'{organism}_dict.csv' in os.listdir(f'{self.file_paths["pathway_xml_files"]}'):
            logging.info(f'\t\t\tReading {organism} dictionary file...')
            with open(pathway_dict_path, 'r') as kegg_dict_file:
                for line in kegg_dict_file:
                    line = line.strip().split('\t')
                    k = line[0]
                    name = line[1]
                    orgDict[k] = name

        # If the dictionary file does not exist, write it and stream in the data for the dictionary
        else:
            logging.info(f'\t\t\tOrganism dictionary not present for {organism}, downloading...')
            try:  # try to retrieve and parse the dictionary containing organism gene names to codes conversion
                url = requests.get("http://rest.kegg.jp/list/" + organism, stream=True)
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
                                aliasDict[nameline[entry].strip()] = name.upper()
                        orgDict[k] = name
                        kegg_dict_file.write(f'{k}\t{name}\n')
                url.close()
            except:
                logging.info("Could not get library: " + organism)

        # Writes xml files for the pathways in the pathway list
        self.write_xml_files(organism, kegg_pathway_list)

        xml_file_path = os.listdir(f'{self.file_paths["pathway_xml_files"]}/{organism}')

        def parse_xml_files(xml_file, pathway_name):
            """
            Reads in the pathway xml file and parses the connections. Creates a networkx directed graph of the pathway
            """
            with open(f'{self.file_paths["pathway_xml_files"]}/{organism}/{xml_file}', 'r') as pathway_file:
                text = [line for line in pathway_file]

                # Read the kegg xml file
                graph = self.read_kegg(text, nx.DiGraph(), kegg_dict, orgDict)

                # Parse the kegg pathway and determine if there is sufficient overlap for processing with scBONITA
                pathway_code = xml_file.replace(".xml", "")
                self.parse_kegg_pathway(graph, minimumOverlap, pathway_code, pathway_num, num_pathways)

        # If there aren't any kegg pathways specified, look for all overlapping pathways
        if len(kegg_pathway_list) == 0:
            # Read in the pre-downloaded xml files and read them into a DiGraph object
            num_pathways = len(xml_file_path)
            logging.info(f'\t\tNo KEGG pathways specified, searching all overlapping pathways')
            logging.info(f'\t\tFinding pathways with at least {minimumOverlap} genes that overlap with the dataset')
            with alive_bar(num_pathways) as bar:
                for pathway_num, xml_file in enumerate(xml_file_path):
                    
                    pathway_name = xml_file.split('.')[0]
                    if f"{pathway_name}.graphml" not in os.listdir(self.output_path):
                        parse_xml_files(xml_file, pathway_name)   # pass both args
                    bar()

        # If there are pathways specified by the user, load those in
        else:
            pathway_list = list(kegg_pathway_list)
            num_pathways = len(pathway_list)
            minimumOverlap = 1  # Minimum number of genes that need to be in both the dataset and pathway for the pathway to be considered
            logging.info(f'\t\tFinding pathways with at least {minimumOverlap} genes that overlap with the dataset')

            with alive_bar(num_pathways) as bar:
                for pathway_num, pathway in enumerate(pathway_list):
                    for xml_pathway_name in xml_file_path:
                        if organism + pathway + '.xml' == xml_pathway_name:
                            parse_xml_files(xml_pathway_name, pathway)

                        elif 'ko' + pathway + '.xml'== xml_pathway_name:
                            parse_xml_files(xml_pathway_name, pathway)
                        
                    bar()

        if len(self.pathway_dict.keys()) == 0:
            raise Exception(f'WARNING: No pathways passed the minimum overlap of {minimumOverlap}')
        
        return self.pathway_dict

    def add_pathways(self, pathway_list, minOverlap, write_graphml=True, removeSelfEdges=False, organism='hsa'):
        """
        Add a list of pathways in graphml format to the rule_inference object

        Writes out the "_processed.graphml" files
        """

        logging.info(f'\t\tAdding graphml pathways to rule_inference object...')

        # Get a list of the genes in the dataset
        if hasattr(self, "cv_genes"):
            pathway_genes = set(self.cv_genes)
        elif not hasattr(self, "cv_genes"):
            pathway_genes = set(self.gene_list)

        def create_processed_networkx_graphml(G, pathway):
            """
            Reads in the graph and the pathway and filters out self edges and isolates

            Creates the "_processed.graphml" files
            """
            nodes = set(G.nodes())

            # Compute the number of nodes that overlap with the pathway genes
            overlap = len(nodes.intersection(pathway_genes))

            # Check to see if there are enough genes in the dataset that overlap with the genes in the pathway
            if overlap >= minOverlap:

                logging.info(f'\t\tPathway: {pathway} Overlap: {overlap} Edges: {len(G.edges())}')
                nodes = list(G.nodes())

                if removeSelfEdges:
                    G.remove_edges_from(nx.selfloop_edges(G))  # remove self loops
                # remove genes not in dataset
                for pg in list(G.nodes()):
                    if pg not in pathway_genes:
                        G.remove_node(pg)


                # graph post-processing
                # remove singletons/isolates
                G.remove_nodes_from(list(nx.isolates(G)))

                self.pathway_graphs[pathway] = G
                logging.info(f'\t\t\tEdges after processing: {len(G.edges())} Overlap: {len(set(G.nodes()).intersection(pathway_genes))}')
                filtered_overlap = len(set(G.nodes()).intersection(pathway_genes))

                if write_graphml and filtered_overlap > minOverlap:
                    base = self.output_path
                    fname = pathway
                    if not fname.startswith(organism):
                        fname = f"{organism}{fname}"
                    if not fname.endswith("_processed.graphml"):
                        fname = f"{fname}_processed.graphml"
                    out_path = os.path.join(base, fname)
                    nx.write_graphml(G, out_path, infer_numeric_types=True)

            else:
                msg = f'Overlap {overlap} is below the minimum {minOverlap}'
                raise Exception(msg)

        # Create the "_processed.graphml" files

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
        
    def build_global_network(self, pathway_dict, write_graphml=True, filename="all_kegg_pathways.graphml"):
        """
        Combine every pathway graph currently in `self.pathway_dict`
        into a single directed graph and (optionally) write it to disk.
        """
        if not pathway_dict:
            raise ValueError("`pathway_dict` is empty – run `find_kegg_pathways` first")

        # Union of all edges/nodes (attributes are preserved; duplicates collapsed)
        G_global = nx.compose_all(pathway_dict.values())

        # Tag every edge with the pathway(s) it came from
        for path_code, g in pathway_dict.items():
            for u, v, data in g.edges(data=True):
                if (u, v) in G_global.edges:
                    G_global[u][v].setdefault("pathways", set()).add(path_code)

        # Write the graph to disk (after converting unsupported types)
        if write_graphml:
            for u, v, d in G_global.edges(data=True):
                if isinstance(d.get("pathways"), set):
                    d["pathways"] = ",".join(sorted(d["pathways"]))

            out_path = os.path.join(self.output_path, filename)
            nx.write_graphml(G_global, out_path, infer_numeric_types=True)
            logging.info(f"Wrote merged network with {G_global.number_of_nodes()} nodes "
                        f"and {G_global.number_of_edges()} edges → {out_path}")

        return G_global
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    base_dir = ROOT_DIR
    kegg_dir = DATA_DIR / "kegg_pathways"
    data_dir = SAMPLE_PROCESSED_DATA_DIR
    
    os.makedirs(kegg_dir, exist_ok=True)
    
    # 0. Instantiate
    logging.info("Creating Pathways object")
    pw = Pathways(
        dataset_name      = DATASET_NAME,
        cv_threshold      = None,
        output_path       = kegg_dir,
        data_file         = "expression_matrix.csv",
        gene_name_file    = COMMON_DATA / "total_genes.csv",
        write_graphml     = True,         # per‑pathway files optional
        organism          = "mmu"
    )

    # 1. Discover *all* organism pathways
    logging.info("Finding all KEGG pathways")
    pathway_dict = pw.find_kegg_pathways(
        kegg_pathway_list = [],            # empty means grab every pathway XML you have/can fetch
        write_graphml     = True,         # skip per‑pathway output if you like
        minimumOverlap    = 1              # keep anything with ≥1 gene from your dataset
    )

    # 2. Merge into one global network
    logging.info("Building global network")
    global_net = pw.build_global_network(
        pathway_dict,
        write_graphml=True,
        filename="all_kegg_pathways.graphml"
    )