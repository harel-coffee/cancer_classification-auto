import pandas as pd
import numpy as np
from . import raw_TCGA_path, processed_TCGA_path
from .dataset import Dataset


raw_clinical_path = raw_TCGA_path / "clinical"

def load_raw_clinical_cohort(cohort_name):
    c = pd.read_table(raw_clinical_path / (cohort_name.upper() + ".gz"), index_col=0)
    c.index.name = "sample"
    c.columns.name = "attribute"
    columns_ordered = sorted(c.columns)
    c = c[columns_ordered]
    c = c.sort_index()
    return c


def load_clinical_cohort(cohort_name):
    c = pd.read_table(processed_TCGA_path / cohort_name / "clinical.tsv", index_col=0)
    
#     idx_to_sample = pd.Series(index=np.arange(c.shape[0], dtype=int), data=c.index.values)
#     sample_to_idx = pd.Series(data=np.arange(c.shape[0], dtype=int), index=c.index.values)
    
#     idx_to_column = pd.Series(index=np.arange(c.shape[1], dtype=int), data=c.columns.values)
#     column_to_idx = pd.Series(data=np.arange(c.shape[1], dtype=int), index=c.columns.values)
    return c