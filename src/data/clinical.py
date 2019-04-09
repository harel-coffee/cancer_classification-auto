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


# ------------------ STAGING  ------------------ #

stage_name_homogeneous = {
    
    np.nan:          (np.nan, np.nan, np.nan),      # UNKNOWN
    '[DISCREPANCY]': (np.nan, np.nan, np.nan),      # UNKNOWN
    'STAGE X':       (np.nan, np.nan, np.nan),      # UNKNOWN
    '[UNKNOWN]':     (np.nan, np.nan, np.nan),      # UNKNOWN
    
    'STAGE 0': (0,np.nan, np.nan),   # in-situ
    'IS':      (0,np.nan, np.nan),   # in-situ
    'STAGE IS':(0,np.nan, np.nan),   # in-situ
    
    'STAGE I': (1,np.nan, np.nan),   # Stage 1
    'I':       (1,np.nan, np.nan),   # Stage 1
    'STAGE IA': (1, "A", np.nan),    # Stage 1
    'STAGE IA1':(1, "A", 1),         # Stage 1
    'STAGE IA2':(1, "A", 2),         # Stage 1
    'STAGE IB': (1, "B", np.nan),    # Stage 1
    'STAGE IB1':(1, "B", 1),         # Stage 1
    'STAGE IB2':(1, "B", 2),         # Stage 1
    'STAGE IC': (1, "C", np.nan),    # Stage 1
    
    'STAGE II': (2, np.nan, np.nan), # Stage 2
    'STAGE I/II (NOS)':(2, np.nan, np.nan), # Stage 2
    'I/II NOS': (2, np.nan, np.nan), # Stage 2
    'STAGE IIA':(2, "A", np.nan),    # Stage 2
    'IIA':      (2, "A", np.nan),    # Stage 2
    'STAGE IIA1':(2, "A", 1),        # Stage 2
    'STAGE IIA2':(2, "A", 2),        # Stage 2
    'STAGE IIB':(2, "B", np.nan),    # Stage 2
    'IIB':      (2, "B", np.nan),    # Stage 2
    'STAGE IIC':(2, "C", np.nan),    # Stage 2
    
    'STAGE III':(3, np.nan, np.nan), # Stage 3
    'III':      (3, np.nan, np.nan), # Stage 3
    'STAGE IIIA':(3, "A", np.nan),   # Stage 3
    'STAGE IIIB':(3, "B", np.nan),   # Stage 3
    'STAGE IIIC':(3, "C", np.nan),   # Stage 3
    'STAGE IIIC1':(3, "C", 1),       # Stage 3
    'STAGE IIIC2':(3, "C", 2),       # Stage 3
    
    'STAGE IV':(4, np.nan, np.nan),  # Stage 4
    'STAGE IVA':(4, "A", np.nan),    # Stage 4
    'IVA':      (4, "A", np.nan),    # Stage 4
    'STAGE IVB':(4, "B", np.nan),    # Stage 4
    'IVB':      (4, "B", np.nan),    # Stage 4,
    'STAGE IVC':(4, "C", np.nan),    # Stage 4
}


def convert_stage_name(name):
    return stage_name_homogeneous[name] 


# ---------------------------------------------- #
# ------------------ GRADING  ------------------ #

HISTOLOGICAL_GRADE = 'histological_grade'

# ---------------------------------------------- #