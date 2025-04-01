import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import stringdb
import mygene

import requests ## python -m pip install requests

def map_gene_identifiers(gene_list):
    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv"
    method = "get_string_ids"

    # Set the parameters
    params = {
        "identifiers" : "\r".join(["p53", "BRCA1", "cdk2", "Q99835"]), # your protein list
        "species" : 9606, # NCBI/STRING taxon identifier 
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 1, # see your input identifiers in the output
        "caller_identity" : "www.awesome_app.org" # your app name
    }

    # Construct the URL
    request_url = "/".join([string_api_url, output_format, method])

    results = requests.post(request_url, data=params)
    
    input_id_list = []
    string_id_list = []
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        input_identifier, string_identifier = l[0], l[2]
        
        input_id_list.append(input_identifier)
        string_id_list.append(string_identifier)
    
    string_id_df = pd.DataFrame({"gene_id": input_id_list, "string_id": string_id_list})
    
    return string_id_df
        

def find_interaction_partners(gene_list, interaction_limit):
    """ interaction_partners API method provides the interactions between your set of proteins and all the other STRING proteins"""
    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv"
    method = "interaction_partners"

    # Construct the request
    request_url = "/".join([string_api_url, output_format, method])

    # Set parameters
    params = {
        "identifiers" : "%0d".join(gene_list), # your protein
        "species" : 9606, # NCBI/STRING taxon identifier 
        "limit" : 5,
        "caller_identity" : "www.awesome_app.org" # your app name

    }

    query_string_id_list = []
    query_prot_name_list = []
    
    partner_string_id_list = []
    partner_prot_name_list = []
    
    #Call STRING
    response = requests.post(request_url, data=params)
    
    for line in response.text.strip().split("\n")[1:]:
        l = line.strip().split("\t")
        query_ensp = l[0]
        query_name = l[2]
        partner_ensp = l[1]
        partner_name = l[3]
        combined_score = l[5]
        
        query_string_id_list.append(query_ensp)
        query_prot_name_list.append(query_name)
        partner_string_id_list.append(partner_ensp)
        partner_prot_name_list.append(partner_name)
                
    interaction_df = pd.DataFrame({
        "string_id": query_string_id_list,
        "query_protein": query_prot_name_list,
        "partner_id": partner_string_id_list,
        "partner_protein": partner_prot_name_list
        })
    
    return interaction_df

def convert_protein_to_gene(df):
    mg = mygene.MyGeneInfo()
    
    
    # Get all unique Ensembl protein IDs from both columns
    all_ids = pd.concat([df["string_id"], df["partner_id"]]).unique().tolist()
    
    all_ids_stripped = [i.strip("9606.") for i in all_ids]

    # Query mygene to map Ensembl protein IDs (using scope "ensembl.protein") to gene symbols
    results = mg.querymany(all_ids_stripped, scopes="ensembl.protein", fields="symbol", species="human")

    # Build a mapping dictionary: key = Ensembl protein ID, value = gene symbol
    mapping_dict = {item["query"]: item.get("symbol", None) for item in results}

    # Create new columns with the mapped gene symbols
    df["query_gene"] = df["string_id"].map(mapping_dict)
    df["partner_gene"] = df["partner_id"].map(mapping_dict)

    # Display the updated DataFrame
    print(df.head())
    
    return df
    
    
gene_list = ["p53", "BRCA1", "cdk2", "Q99835"]
string_id_df = map_gene_identifiers(gene_list)

mapped_gene_list = string_id_df["string_id"].to_list()
interaction_df = find_interaction_partners(mapped_gene_list, 5)

print(interaction_df.head())

string_gene_interactions = convert_protein_to_gene(interaction_df)





