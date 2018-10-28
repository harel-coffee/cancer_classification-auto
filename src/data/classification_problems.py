import pandas as pd
import numpy as np
from .genexp import load_genexp_cohort
from .clinical import load_clinical_cohort


def load_sample_classification_problem(cohort_name):
    
    classes_mapping = {
        'Primary Tumor': 1,
        'Solid Tissue Normal': 0
    }
    
    cohort_clinical = load_clinical_cohort(cohort_name)
    X, idx_to_sample, sample_to_idx = load_genexp_cohort(cohort_name)
    clinical_with_class = cohort_clinical[cohort_clinical.sample_type.isin(classes_mapping.keys())]
    
    # filtering X
    samples_with_class = clinical_with_class.index.tolist()
    sample_to_idx_filtered = sample_to_idx[samples_with_class].values
    
    new_sample_to_idx = pd.Series(index=samples_with_class, data=np.arange(len(samples_with_class), dtype=int))
    new_idx_to_sample = pd.Series(data=samples_with_class, index=np.arange(len(samples_with_class), dtype=int))
    
    X_filtered = X[sample_to_idx_filtered, :]
    
    # creation of y
    y = clinical_with_class.sample_type.map(lambda x: classes_mapping[x]).astype(int).values
    
    return X_filtered, y, new_idx_to_sample, new_sample_to_idx
    
    