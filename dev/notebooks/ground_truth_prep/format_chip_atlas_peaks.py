import os
import pandas as pd
import pybedtools

ground_truth_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files"

dataset_name = "K562"
chip_atlast_file_name = "Oth.Bld.05.AllAg.K-562.bed"
chip_atlas_bed_file = os.path.join(ground_truth_dir, chip_atlast_file_name)
formatted_path = os.path.join(ground_truth_dir, f"chipatlas_{dataset_name}.csv")

tss_bed_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/genome_annotation/hg38/gene_tss.bed"
genome_file = genome_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/reference_genome/hg38/hg38.chrom.sizes"


def map_chip_atlas_peaks_to_closest_tss(
    chip_atlas_bed_file: str,
    tss_bed_file: str,
    genome_file: str
) -> pd.DataFrame:
    """
    Map Chip-Atlas peaks to their closest gene TSS within a specified distance cutoff.
    
    Parameters
    ----------
    chip_atlas_bed_file : str
        Path to the Bed file containing Chip-Atlas peaks.
    tss_bed_file : str
        Path to the Bed file containing gene TSS locations.
    genome_file : str
        Path to the genome file for sorting BedTools objects.
        
    Returns
    -------
    pd.DataFrame : pd.DataFrame
        DataFrame containing peak-gene pairs with their distances.
    """
    chip_bed = pybedtools.BedTool(chip_atlas_bed_file)
    tss_bed = pybedtools.BedTool(tss_bed_file)

    # drop random/chrUn contigs
    tss_bed = tss_bed.filter(lambda f: "random" not in f.chrom and "chrUn" not in f.chrom).saveas()

    # Sort both consistently
    if genome_file:
        chip_sorted = chip_bed.sort(g=genome_file)
        tss_sorted  = tss_bed.sort(g=genome_file)
    else:
        chip_sorted = chip_bed.sort()
        tss_sorted  = tss_bed.sort()

    # Now run closest on the sorted objects
    chip_closest_tss = chip_sorted.closest(tss_sorted, d=True)
    
    raw_chip_closest_tss_df = chip_closest_tss.to_dataframe(
    names=[
            # fields from the ChIP peak AFTER chrom/start/end:
            "peak_name",
            "peak_score",
            "peak_strand",
            "peak_thick_start",
            "peak_thick_end",
            "peak_rgb",
            "tss_chr",
            "tss_start",
            "tss_end",
            "tss_gene",
            "distance"
        ]
    ).reset_index()
    
    raw_chip_closest_tss_df = raw_chip_closest_tss_df.rename(
    columns={
            "level_0": "peak_chr",
            "level_1": "peak_start",
            "level_2": "peak_end",
        }
    )
    raw_chip_closest_tss_df["source_id"] = (
    raw_chip_closest_tss_df["peak_name"]
        .astype(str)
        .str.extract(r"Name=([^%]+)", expand=False)
        .str.upper()
    )

    raw_chip_closest_tss_df["peak_id"] = (
        raw_chip_closest_tss_df["peak_chr"].astype(str)
        + ":" +
        raw_chip_closest_tss_df["peak_start"].astype(str)
        + "-" +
        raw_chip_closest_tss_df["peak_end"].astype(str)
    )
    
    raw_chip_closest_tss_df["source_id"] = raw_chip_closest_tss_df["source_id"].str.upper()
    raw_chip_closest_tss_df["target_id"] = raw_chip_closest_tss_df["tss_gene"].str.upper()

    chip_closest_tss_df = raw_chip_closest_tss_df[["source_id", "peak_id", "target_id", "distance"]]
    
    return chip_closest_tss_df

chip_atlas_tf_peak_tg_dist = map_chip_atlas_peaks_to_closest_tss(
    chip_atlas_bed_file=chip_atlas_bed_file,
    tss_bed_file=tss_bed_file,
    genome_file=genome_file
    )

chip_atlas_tf_peak_tg_dist.to_csv(formatted_path, index=False)
print(f"Formatted Chip-Atlas peak to gene TSS distances saved to: {formatted_path}")