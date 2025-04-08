import pandas as pd
import mygene
from tqdm import tqdm

human_net_db_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/human_net_db/human_net_plus_entrez.tsv"

print("Reading the HumanNet database")
human_net_db = pd.read_csv(human_net_db_file, sep="\t", header=None)

# Initialize the MyGeneInfo client
mg = mygene.MyGeneInfo()

# List of Entrez gene IDs as strings (or integers)
col0_list = human_net_db.loc[:, 0].to_list()
col1_list = human_net_db.loc[:, 1].to_list()

def entrez_id_to_gene_symbol(entrez_id_list: list, species: str):
    # Query MyGene.info, specifying the scopes (i.e. the source) and fields you wish to retrieve
    results = mg.querymany(entrez_id_list, scopes="entrezgene", fields="symbol", species=species)

    # Process the results to get a mapping from Entrez ID to gene symbol
    id_to_symbol = {}
    for entry in tqdm(results):
        # entry might contain a 'notfound' key if there's no match
        if "notfound" in entry and entry["notfound"]:
            continue
        else:
            id_to_symbol[entry["query"]] = entry.get("symbol", "NA")
    
    return id_to_symbol

print("Converting geneA Entrez IDs to gene symbols")
geneA_dict = entrez_id_to_gene_symbol(col0_list, species="human")
human_net_db[0] = human_net_db[0].astype(str)
human_net_db["geneA"] = human_net_db[0].map(geneA_dict)

print("Converting geneB Entrez IDs to gene symbols")
geneB_dict = entrez_id_to_gene_symbol(col1_list, species="human")
human_net_db[1] = human_net_db[1].astype(str)
human_net_db["geneB"] = human_net_db[1].map(geneB_dict)

human_net_db = human_net_db.rename(columns={
    human_net_db.columns[2]: "human_net_score",
    })

human_net_db = human_net_db[["geneA", "geneB", "human_net_score"]]

print("Writing to 'human_net_db/human_net_gene_name_db.csv'")
human_net_db.to_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/human_net_db/human_net_gene_name_db.csv", header=True)

