#!/bin/bash -l
#SBATCH --job-name="download_mesc_raw_data"
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/download_raw_mesc_files.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/download_raw_mesc_files.err

set -euo pipefail

E7_5_rep1_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205427/suppl/GSM6205427%5FE7.5%5Frep1%5FATAC%5Ffragments.tsv.gz"
)

E7_5_rep2_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fmatrix.mtx.gz"
    ""https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205428/suppl/GSM6205428%5FE7.5%5Frep2%5FATAC%5Ffragments.tsv.gz""
)

E7_75_rep1_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205429/suppl/GSM6205429%5FE7.75%5Frep1%5FATAC%5Ffragments.tsv.gz"
)

E8_0_rep1_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205430/suppl/GSM6205430%5FE8.0%5Frep1%5FATAC%5Ffragments.tsv.gz"
)

E8_0_rep2_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205431/suppl/GSM6205431%5FE8.0%5Frep2%5FATAC%5Ffragments.tsv.gz"
)

E8_5_CRISPR_T_KO_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205432/suppl/GSM6205432%5FE8.5%5FCRISPR%5FT%5FKO%5FATAC%5Ffragments.tsv.gz"
)

E8_5_CRISPR_T_WT_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205433/suppl/GSM6205433%5FE8.5%5FCRISPR%5FT%5FWT%5FATAC%5Ffragments.tsv.gz"
)

E8_5_rep1_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205434/suppl/GSM6205434%5FE8.5%5Frep1%5FATAC%5Ffragments.tsv.gz"
)

E8_5_rep2_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205435/suppl/GSM6205435%5FE8.5%5Frep2%5FATAC%5Ffragments.tsv.gz"
)

E8_75_rep1_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205436/suppl/GSM6205436%5FE8.75%5Frep1%5FATAC%5Ffragments.tsv.gz"
)

E8_75_rep2_links=(
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fbarcodes.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Ffeatures.tsv.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fmatrix.mtx.gz"
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205437/suppl/GSM6205437%5FE8.75%5Frep2%5FATAC%5Ffragments.tsv.gz"
)

declare -A samples
samples=( 
    ["E7.5_rep1"]=E7_5_rep1_links
    ["E7.5_rep2"]=E7_5_rep2_links
    ["E7.75_rep1"]=E7_75_rep1_links
    ["E8.0_rep1"]=E8_0_rep1_links
    ["E8.0_rep2"]=E8_0_rep2_links
    ["E8.5_CRISPR_T_KO"]=E8_5_CRISPR_T_KO_links
    ["E8.5_CRISPR_T_WT"]=E8_5_CRISPR_T_WT_links
    ["E8.5_rep1"]=E8_5_rep1_links
    ["E8.5_rep2"]=E8_5_rep2_links
    ["E8.75_rep1"]=E8_75_rep1_links
    ["E8.75_rep2"]=E8_75_rep2_links
    )

DOWNLOAD_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES"

mkdir -p $DOWNLOAD_DIR

for sample_label in "${!samples[@]}"; do
    array_name="${samples[$sample_label]}"
    declare -n links="$array_name"

    sample_dir="${DOWNLOAD_DIR}/${sample_label}"
    mkdir -p "$sample_dir"

    echo "Downloading: $sample_label"
    for file_url in "${links[@]}"; do
        fname=$(basename "$file_url" | sed 's/%5F/_/g')
        outpath="${sample_dir}/${fname}"

        if [[ -f "$outpath" ]]; then
            echo "  - $fname exists, skipping"
        else
            echo "  - Downloading: $fname"
            wget -q -O "$outpath" "$file_url"
        fi
    done
    echo ""
done

echo "DONE"

