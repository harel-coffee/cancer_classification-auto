import pandas as pd
import numpy as np
from . import raw_TCGA_path, processed_TCGA_path

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
    X = np.load(processed_TCGA_path / cohort_name.upper() / "X.npy")
    idx_to_sample = __load_index(processed_TCGA_path / cohort_name.upper() / "idx_to_sample.tsv")
    sample_to_idx = __load_index(processed_TCGA_path / cohort_name.upper() / "sample_to_idx.tsv")
    return X, idx_to_sample, sample_to_idx