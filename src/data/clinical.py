import pandas as pd
from . import raw_TCGA_path, processed_TCGA_path

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
    return c