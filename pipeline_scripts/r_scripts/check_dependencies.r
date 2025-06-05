# =============================================
# Install all required packages (CRAN + Bioconductor + GitHub)
# =============================================

# Required packages
required_cran_packages <- c(
  "Matrix", "reshape2", "dplyr", "tidyr", "parallel", 
  "arrow", "terra", "ggrastr", "devtools", "lme4", "limma"
)

required_bioc_packages <- c(
  "BiocGenerics", "DelayedArray", "DelayedMatrixStats", "S4Vectors",
  "SingleCellExperiment", "SummarizedExperiment", "batchelor",
  "HDF5Array", "GenomicRanges", "rtracklayer", "Signac", "cicero"
)

# ---------------------------------------------
# Install CRAN packages
# ---------------------------------------------
install_missing_cran <- function(pkgs) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing missing CRAN package: %s\n", pkg))
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}

# ---------------------------------------------
# Install Bioconductor packages
# ---------------------------------------------
install_missing_bioc <- function(pkgs) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install(version = "3.20", ask = FALSE)

  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing missing Bioconductor package: %s\n", pkg))
      BiocManager::install(pkg, ask = FALSE)
    }
  }
}

# ---------------------------------------------
# Install monocle3 from GitHub (cole-trapnell-lab)
# ---------------------------------------------
install_monocle3_github <- function() {
  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org")
  }
  cat("Installing monocle3 from GitHub (cole-trapnell-lab)...\n")
  devtools::install_github("cole-trapnell-lab/monocle3", ref = "master", upgrade = "never")
}

# =============================================
# Run installations
# =============================================
cat("Installing CRAN packages...\n")
install_missing_cran(required_cran_packages)

cat("Installing Bioconductor packages...\n")
install_missing_bioc(required_bioc_packages)

cat("Installing monocle3 from GitHub...\n")
install_monocle3_github()

cat("All dependencies installed successfully.\n")
