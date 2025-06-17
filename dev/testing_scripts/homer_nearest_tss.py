import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Tuple

homer_tf_motif_score_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/DS011_mESC/DS011_mESC_sample1/homer_results/homer_tf_motif_scores"
ground_truth_path = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"

def read_ground_truth(ground_truth_file):
    """Read ground truth TF--target pairs from a tab-delimited file.

    Parameters
    ----------
    ground_truth_file : str
        File path to a TSV with columns ``Source`` and ``Target``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column names ``source_id`` and ``target_id``.
    """
    ground_truth = pd.read_csv(
        ground_truth_file,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        header=0
    )
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    ground_truth = ground_truth[["source_id", "target_id"]]
    
    ground_truth["source_id"] = ground_truth["source_id"].str.upper()
    ground_truth["target_id"] = ground_truth["target_id"].str.upper()
    
    return ground_truth

ground_truth_df = read_ground_truth(ground_truth_path)
ground_truth_tfs = set(ground_truth_df["source_id"].unique())

tf_targets: list[pd.DataFrame] = []
for known_motif_file in tqdm(os.listdir(homer_tf_motif_score_dir)):
    motif_path = os.path.join(homer_tf_motif_score_dir, known_motif_file)
    
    # Only read the header row to get column names
    try:
        with open(motif_path, "r") as f:
            header = f.readline().strip().split("\t")
            TF_column = header[-1]
    except Exception as e:
        print(f"Error reading file {known_motif_file}: {e}")
        continue

    # Extract TF name robustly
    TF_name = TF_column.split('/')[0].split('(')[0].split(':')[0].upper()

    if TF_name in ground_truth_tfs:    
        print(TF_name)

        homer_df = pd.read_csv(
            motif_path, 
            sep="\t", 
            header=0, 
            index_col=0 # Setting the PeakID column as the index
            )
        homer_df.index.names = ["PeakID"]
        
        homer_df["peak_id"] = homer_df["Chr"].astype(str) + ":" + homer_df["Start"].astype(str) + "-" + homer_df["End"].astype(str)
        homer_df["source_id"] = TF_name
        homer_df = homer_df.rename(columns={"Gene Name":"target_id"})
        
        cols_of_interest = [
            "source_id",
            "peak_id",
            "target_id",
            "Chr",
            "Start",
            "End",
            "Annotation",
            "Distance to TSS",
            "Gene Type",
            "CpG%",
            "GC%"
        ]
        homer_df = homer_df[cols_of_interest]
        homer_df["target_id"] = homer_df["target_id"].str.upper()

        valid_targets = set(ground_truth_df["target_id"]) & set(homer_df["target_id"])
        valid_sources = set(ground_truth_df["source_id"]) & set(homer_df["source_id"])
        
        ground_truth_tf_edges = ground_truth_df[ground_truth_df["source_id"] == TF_name]

        homer_shared_genes = homer_df[
            homer_df["source_id"].isin(valid_sources) &
            homer_df["target_id"].isin(valid_targets)
        ].reset_index(drop=True)
        
        ground_truth_shared_genes = ground_truth_tf_edges[
            ground_truth_tf_edges["source_id"].isin(valid_sources) &
            ground_truth_tf_edges["target_id"].isin(valid_targets)
        ].reset_index(drop=True)
                
        tf_tg_overlap_w_ground_truth = pd.merge(
            ground_truth_shared_genes,
            homer_shared_genes,
            on=["source_id", "target_id"],
            how="outer",
            indicator=True
        )
        
        tf_targets.append(tf_tg_overlap_w_ground_truth)
    
total_tf_to_tg = pd.concat(tf_targets)
print(total_tf_to_tg)
print(total_tf_to_tg.shape)

overlapping_edges = total_tf_to_tg[total_tf_to_tg["_merge"] == "both"]
edges_only_in_homer = total_tf_to_tg[total_tf_to_tg["_merge"] == "right_only"]
edges_only_in_ground_truth = total_tf_to_tg[total_tf_to_tg["_merge"] == "left_only"]

print(f"Number of edges in both Homer and ground truth: {len(overlapping_edges):,.0f}")
print(f"Number of edges only in Homer: {len(edges_only_in_homer):,.0f}")
print(f"Number of edges only in the ground truth: {len(edges_only_in_ground_truth):,.0f}")

tp = (total_tf_to_tg["_merge"] == "both").sum()
fn = (total_tf_to_tg["_merge"] == "left_only").sum()
fp = (total_tf_to_tg["_merge"] == "right_only").sum()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Recall = {recall:.3f} ({tp} / {tp+fn})")
 


tss_reference = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_annotation/mm10/mm10_TSS.bed", 
    sep="\t", 
    header=None, 
    index_col=None,
    names=["Chr", "Start", "End", "Name", "Strand", "Strand2"]
    )
tss_reference["tss"] = (tss_reference["Start"] + tss_reference["End"]) // 2

gene_body_anno = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_annotation/mm10/mm10_gene_body_anno.bed", 
    sep="\t", 
    header=None, 
    index_col=None,
    names=["Chr", "Start", "End", "Name", "Strand", "Strand2"]
    )

def plot_gene_enhancers(df, gene, tss_reference, gene_body_anno, figsize=(10, 2)):
    gene_df = df[df["target_id"] == gene].copy()
    if gene_df.empty:
        print(f"No enhancers found for {gene}")
        return
    
    gene_df = gene_df.drop_duplicates(subset="peak_id")
    gene_df["peak_center"] = (gene_df["Start"] + gene_df["End"]) // 2
    
    def find_target_gene_tss(gene) -> Tuple[float, float]:
        """Standardizes the target gene TSS using a reference dataset"""
        tss_entry = tss_reference[tss_reference["Name"].str.upper() == gene.upper()]
        if tss_entry.empty:
            raise ValueError(f"Gene '{gene}' not found in TSS reference.")
        tss_start = float(tss_entry["Start"].iloc[0])
        tss_end = float(tss_entry["End"].iloc[0])
        return tss_start, tss_end
    
    def find_target_gene_body(gene) -> Tuple[float, float]:
        """Standardizes the target gene length using a reference dataset"""
        gene_entry = gene_body_anno[gene_body_anno["Name"].str.upper() == gene.upper()]
        if gene_entry.empty:
            raise ValueError(f"Gene '{gene}' not found in gene body reference")
        gene_start = float(gene_entry["Start"].iloc[0])
        gene_end = float(gene_entry["End"].iloc[0])
        return gene_start, gene_end
    
    gene_start, gene_end = find_target_gene_body(gene)
    tss_start, tss_end = find_target_gene_tss(gene)

    # Set a distance cutoff so the figure doesn't get too squished
    max_dist = 250000
    gene_df["distance_to_gene"] = abs(gene_df["peak_center"] - ((gene_start + gene_end) / 2))
    gene_df = gene_df[gene_df["distance_to_gene"] <= max_dist]
    
    if gene_df.empty:
        print(f"No enhancers within ±{max_dist:,} bp of {gene}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # Blue Rectangle the length of the gene body
    start_height = 25
    gene_height = 10
    
    ax.add_patch(
        patches.Rectangle((gene_start, start_height), gene_end - gene_start, gene_height,
                          facecolor='skyblue', edgecolor='black', label=gene)
    )
    
    if abs(tss_start - gene_start) < abs(tss_start - gene_end):
        tss_rect_start = min(tss_start, gene_start)
        tss_rect_width = abs(tss_start - gene_start)
    else:
        tss_rect_start = min(tss_start, gene_end)
        tss_rect_width = abs(tss_start - gene_end)

    ax.add_patch(
        patches.Rectangle((tss_rect_start, start_height), tss_rect_width, gene_height,
                        facecolor='green', edgecolor='black')
    )

    xmin = int(np.ceil(tss_start - 25000))
    xmax = int(np.ceil(gene_end + 25000))
    xrange = xmax - xmin
    h = xrange * 0.02     # gene height
    arc_height = max(50, xrange * 0.001)
    
    # Add a line along the bottom to show the genomic position
    plt.plot([xmin, xmax], [start_height, start_height], 'k-', lw=2)
    
    # Add 10 evenly spaced marks along the line to show the genomic position
    mark_points = [int(np.ceil(np.mean(i))) for i in np.array_split(range(xmin, xmax), 10)]
    for xloc in mark_points:
        # Plots a vertical dash for each of the 10 points
        plt.plot([xloc, xloc], [15, start_height], 'k-', lw=2)
        # Plots the genomic position for each of the 10 points
        plt.text(xloc, 0, s=f'{xloc}', ha='center', va='bottom', fontsize=10)
    
    for _, row in gene_df.iterrows():
        
        # Plot a grey rectangle for the enhancer locations
        ax.add_patch(
        patches.Rectangle((row["Start"], start_height), row["End"] - row['Start'], gene_height / 2,
                          facecolor='grey', edgecolor='black')
        )
        
        enh_center = row["peak_center"]
        tss = tss_start
        center = (enh_center + tss) / 2
        
        radius = max(abs(enh_center - tss) / 2, xrange * 0.001)  # avoid tiny lines

        arc = patches.Arc((center, start_height + 10), radius * 2, arc_height,
                          angle=0, theta1=0, theta2=180)
        ax.add_patch(arc)

    ax.set_xlim(tss_start - 25000, gene_end + 25000)
    ax.set_ylim(0, 75)
    ax.set_title(f"Enhancers targeting {gene}")
    ax.axis("off")
    
    plt.tight_layout()
    plt.show()

plot_gene_enhancers(homer_df, "KIF19A", tss_reference, gene_body_anno)
