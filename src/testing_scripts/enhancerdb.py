import pandas as pd

def load_enhancer_database_file(enhancer_db_file):
    enhancer_db = pd.read_csv(enhancer_db_file, sep="\t", header=None, index_col=None)
    enhancer_db = enhancer_db.rename(columns={
        0 : "chr",
        1 : "start",
        2 : "end",
        3 : "num_enh",
        4 : "tissue",
        5 : "R1_value",
        6 : "R2_value",
        7 : "R3_value",
        8 : "score"
    })
    
    # Average the score of an enhancer across all tissues / cell types
    enhancer_db = enhancer_db.groupby(["chr", "start", "end", "num_enh"], as_index=False)["score"].mean()
    
    enhancer_db["chr"] = enhancer_db["chr"].str.replace("chr", "")
    enhancer_db["start"] = enhancer_db["start"].astype(int)
    enhancer_db["end"] = enhancer_db["end"].astype(int)

    return enhancer_db

def find_peaks_in_enhancer(enhancer, tree_dict):
    """
    Find peaks whose centers fall within an enhancer interval.
    
    Parameters:
      enhancer (Series): A row from the enhancer DataFrame with columns 'chr', 'start', 'end'
      tree_dict (dict): Dictionary mapping chromosome to (KDTree, index list)
      
    Returns:
      List of indices from the original peak_df that fall within the enhancer interval.
    """
    chrom = enhancer["chr"]
    start = enhancer["start"]
    end = enhancer["end"]
    
    # Check if the chromosome is present in our KDTree dictionary
    if chrom not in tree_dict:
        return []
    
    tree, idx_list = tree_dict[chrom]
    enhancer_center = (start + end) / 2.0
    enhancer_radius = (end - start) / 2.0
    
    # Query the KDTree for all peaks within the enhancer's interval
    indices = tree.query_ball_point(np.array([[enhancer_center]]), r=enhancer_radius)[0]
    return [idx_list[i] for i in indices]