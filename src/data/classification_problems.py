import pandas as pd
import numpy as np
from .genexp import load_genexp_cohort
from .clinical import load_clinical_cohort
from .dataset import Dataset


def load_sample_classification_problem(cohort_name):
    
    classes_mapping = [
        'Solid Tissue Normal',
        'Primary Tumor'
    ]
    
    cohort_clinical = load_clinical_cohort(cohort_name)
    cohort_genexp = load_genexp_cohort(cohort_name)
    clinical_with_class = cohort_clinical.data[cohort_clinical.data.sample_type.isin(classes_mapping)]
    
    # filtering X
    samples_with_class = clinical_with_class.index.tolist()
    sample_to_idx_filtered = cohort_clinical.sample_to_idx[samples_with_class].values
    
    new_sample_to_idx = pd.Series(index=samples_with_class, data=np.arange(len(samples_with_class), dtype=int))
    new_idx_to_sample = pd.Series(data=samples_with_class, index=np.arange(len(samples_with_class), dtype=int))
    
    X_filtered = cohort_genexp.data[sample_to_idx_filtered, :]
    
    # creation of y
    y = clinical_with_class.sample_type.map(lambda x: classes_mapping.index(x)).astype(int).values
    result = Dataset(X_filtered, new_idx_to_sample, new_sample_to_idx, 
                     cohort_genexp.idx_to_column, cohort_genexp.column_to_idx, y, classes_mapping)
    return result
    
    