import pandas as pd
import requests

# Example input DataFrame
df = pd.DataFrame({
    'peak_id': ['chr1:100-200', 'chr2:300-400'],
    'source_id': ['TP53', 'BRCA1'],
    'target_id': ['CDKN1A', 'RAD51']
})

# -------------------------------
# Define functions to query databases
# -------------------------------

TRRUST_HUMAN_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/testing_scripts/trrust_rawdata.human.tsv"
trrust_df = pd.read_csv(TRRUST_HUMAN_FILE, sep="\t", header=None, names=['TF', 'Target', 'Regulation', 'PMID'])
# Assuming TRRUST columns include: 'TF', 'Target', 'Regulation', and 'PMID'

def query_trrust(tf, target):
    """
    Look up evidence in the TRRUST dataset that the transcription factor (tf)
    regulates the target gene. The lookup is case-insensitive.
    """
    tf_upper = tf.upper()
    target_upper = target.upper()
    subset = trrust_df[
        (trrust_df['TF'].str.upper() == tf_upper) &
        (trrust_df['Target'].str.upper() == target_upper)
    ]
    if not subset.empty:
        return 1
    else:
        return 0
    
def query_encode(tf):
    """
    Query ENCODE for ChIP-seq experiments for the given TF.
    This version adds the status=released parameter to ensure that only released experiments are returned.
    """
    tf_query = tf.upper()  # ensure uppercase if that's what ENCODE expects
    url = (
        f"https://www.encodeproject.org/search/"
        f"?type=Experiment&assay_title=ChIP-seq&target.label={tf_query}"
        f"&status=released&format=json&limit=all"
    )
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        experiments = data.get("@graph", [])
        num_experiments = len(experiments)
        return f"ENCODE: {num_experiments} ChIP-seq experiments for {tf_query}"
    except Exception as e:
        return f"ENCODE: Error retrieving data for {tf_query}: {e} (URL: {url})"

def query_hichip(peak_id):
    """
    Query a HiChIP or 3D genome resource for enhancer–promoter interactions overlapping the given peak.
    Here, we simulate a query.
    """
    # In a real-world scenario, you might use an API (or local file search) to find loops overlapping 'peak_id'
    return f"HiChIP: Found interactions overlapping {peak_id}"

def query_vista(peak_id):
    """
    Query the VISTA Enhancer Browser to check for validated enhancer activity at the peak.
    """
    # VISTA data is available via web download or browser; here we simulate a lookup.
    return f"VISTA: Enhancer activity reported at {peak_id}"

# -------------------------------
# Function to pull evidence for each row
# -------------------------------

def pull_evidence(row):
    tf = row['source_id']
    target = row['target_id']
    peak = row['peak_id']
    
    return pd.Series({
        'TRRUST': query_trrust(tf, target),
        'ENCODE': query_encode(tf),
        'BioGRID': query_biogrid(tf),
        'HiChIP': query_hichip(peak),
        'VISTA': query_vista(peak)
    })

# Apply the function to each row and create new columns
evidence_df = df.apply(pull_evidence, axis=1)
df = pd.concat([df, evidence_df], axis=1)

# Display the results
print(df)
