import pandas as pd
import numpy as np
from . import raw_TCGA_path, processed_TCGA_path
from .dataset import Dataset

raw_genexp_path = raw_TCGA_path / "genexp"

def load_raw_genexp_cohort(cohort_name):
    df = pd.read_table(raw_genexp_path / (cohort_name.upper() + ".gz"), index_col=0).T
    df.columns.name = "gene"
    columns_sorted = sorted(df.columns)
    df = df[columns_sorted]
    df.index.name = "sample"
    df = df.sort_index()
    return df


def __load_index(index_path):
    return pd.read_table(index_path, index_col=0, squeeze=True)


def load_genexp_cohort(cohort_name):
    """Given the cohort name, returns the gene expression matrix and the relative row indices
    
    Args:
        cohort_name (string): name of the cohort
    
    Returns:
        dataset
    """
    X = np.load(processed_TCGA_path / cohort_name.upper() / "X.npy")
    idx_to_sample = __load_index(processed_TCGA_path / cohort_name.upper() / "idx_to_sample.tsv")
    sample_to_idx = __load_index(processed_TCGA_path / cohort_name.upper() / "sample_to_idx.tsv")
    
    idx_to_gene, gene_to_idx = load_gene_indices()
    
    return Dataset(X, idx_to_sample, sample_to_idx, idx_to_gene, gene_to_idx)


def load_gene_indices():
    return __load_index(processed_TCGA_path / "idx_to_gene.tsv"), __load_index(processed_TCGA_path / "gene_to_idx.tsv")