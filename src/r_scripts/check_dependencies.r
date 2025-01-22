# Handle installing missing packages
required_packages <- c(
  "cicero", "monocle3", "Signac", "Seurat", "GenomicRanges",
  "Matrix", "rtracklayer", "reshape2", "dplyr", "tidyr", "parallel"
)

install_missing_packages <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing missing package: %s\n", pkg))
      install.packages(pkg, repos = "http://cran.r-project.org")
    }
  }
}

install_missing_bioc_packages <- function(packages) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "http://cran.r-project.org")
  }
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing missing Bioconductor package: %s\n", pkg))
      BiocManager::install(pkg, ask = FALSE)
    }
  }
}

# Install CRAN packages
cran_packages <- setdiff(required_packages, c("cicero", "monocle3", "Signac", "GenomicRanges", "rtracklayer"))
install_missing_packages(cran_packages)

# Install Bioconductor packages
bioc_packages <- intersect(required_packages, c("cicero", "monocle3", "Signac", "GenomicRanges", "rtracklayer"))
install_missing_bioc_packages(bioc_packages)

cat("        All R dependencies are installed.\n")
