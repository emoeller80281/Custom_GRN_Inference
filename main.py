import pandas as pd
from Bio import SeqIO
from Bio import motifs
from Bio.Seq import Seq
import requests

hg38_file = "reference_genome/GRCh38_genomic.fna"

def read_genome(genome_fasta_file):
    # Parse the genome of the fasta file
    return SeqIO.to_dict(SeqIO.parse(genome_fasta_file, "fasta"))

def find_sequence_at_location(genome, location):
    "Retrieves the referenge genome sequence for an ATACseq peak location"
    
    chromosomes = {
        "chr1" : "NC_000001.11",
        "chr2" : "NC_000002.12",
        "chr3" : "NC_000003.12",
        "chr4" : "NC_000004.12",
        "chr5" : "NC_000005.10",
        "chr6" : "NC_000006.12",
        "chr7" : "NC_000007.14",
        "chr8" : "NC_000008.11",
        "chr9" : "NC_000009.12",
        "chr10" : "NC_000010.11",
        "chr11" : "NC_000011.10",
        "chr12" : "NC_000012.12",
        "chr13" : "NC_000013.11",
        "chr14" : "NC_000014.9",
        "chr15" : "NC_000015.10",
        "chr16" : "NC_000016.10",
        "chr17" : "NC_000017.11",
        "chr18" : "NC_000018.10",
        "chr19" : "NC_000019.10",
        "chr20" : "NC_000020.11",
        "chr21" : "NC_000021.9",
        "chr22" : "NC_000022.11",
        "chr23" : "NC_000023.11",
        "chr24" : "NC_000024.10",
    }
    
    chromosome = location.split(':')[0]
    start = location.split(":")[1].split('-')[0]
    end = location.split(":")[1].split('-')[1]

    print(f'Chromsome: {chromosome}, Start {start}, End {end}')
    
    chr_to_nc = chromosomes[chromosome]
    
    return genome[chr_to_nc][int(start):int(end)]


def convert_to_homer_peak_format(atac_data):
    # Extract the peak ID column (assuming the first column contains peak IDs)
    peak_ids = atac_data.iloc[:, 0]  # Adjust column index if peak IDs are not in the first column

    # Split peak IDs into chromosome, start, and end
    chromosomes = peak_ids.str.split(':').str[0]
    starts = peak_ids.str.split(':').str[1].str.split('-').str[0]
    ends = peak_ids.str.split(':').str[1].str.split('-').str[1]

    # Create a dictionary for constructing the HOMER-compatible DataFrame
    homer_dict = {
        "peak_id": ["peak" + str(i + 1) for i in range(len(peak_ids))],  # Generate unique peak IDs
        "chromosome": chromosomes,
        "start": starts,
        "end": ends,
        "strand": ["."] * len(starts),  # Set strand as "."
    }

    # Construct the DataFrame
    homer_df = pd.DataFrame(homer_dict)

    # Convert 'start' and 'end' to numeric types
    homer_df['start'] = pd.to_numeric(homer_df['start'])
    homer_df['end'] = pd.to_numeric(homer_df['end'])

    # Print the head of the resulting DataFrame
    print(homer_df.head())

    return homer_df
    
    
    

# ----- Input -----
# Read in the ATAC data
atac_data = pd.read_csv("input/macrophage_buffer1_filtered_ATAC.csv")
homer_df = convert_to_homer_peak_format(atac_data)

homer_peak_file = homer_df.to_csv("input/Homer_peaks.txt", sep='\t', header=False, index=False)

# Read in the RNA data
rna_data = pd.read_csv("input/macrophage_buffer1_filtered_RNA.csv")

# Read in the reference genome
genome = read_genome(hg38_file)

# Find the sequence at the peak position

# Get the first column

# (First entry in the ATACseq dataset)
first_entry = atac_data.iloc[:, 0][0]

sequence = find_sequence_at_location(genome, first_entry)

print(sequence)




# Standardize the gene names

# ----- Standardization and TF TG Identification -----
# Standardize gene names
# Find sequences from reference genome at peak position
# Determine TFs from TGs in the RNA data
# Find TF binding motifs from RNA data

# ----- Calculating TF binding to enhancers -----
# Find the open enhancers
# Match binding motifs to the enhancers
# Calculate a binding score for each enhancer for each TF

# ----- Motif Enrichment -----

