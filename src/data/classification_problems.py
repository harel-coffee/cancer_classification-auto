import pandas as pd
import numpy as np
from .genexp import load_genexp_cohort
from .clinical import load_clinical_cohort, convert_stage_name
from .dataset import Dataset
from . import TCGA_COHORTS, external_data_path
import logging


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


def get_normal_vs_tumor_task(cohort_name):
    clinical = load_clinical_cohort(cohort_name)
    X = load_genexp_cohort(cohort_name)
    
    classes = [
        'Solid Tissue Normal',
        'Primary Tumor'
    ]
    
    samples_with_class = clinical[clinical.sample_type.isin(classes)].index.tolist()
    X_filtered = X.loc[samples_with_class]
    y = clinical.loc[samples_with_class].sample_type.map(lambda x: classes.index(x)).astype(int).values
    idx_to_sample = pd.Series(data = X_filtered.index, index=np.arange(X_filtered.shape[0], dtype=int))
    idx_to_gene = pd.Series(data = X_filtered.columns, index=np.arange(X_filtered.shape[1], dtype=int))
    return X_filtered.values, idx_to_sample, idx_to_gene, y, classes
    
    
def get_stage_classification_task():
    all_clinical = []
    all_genexp = []
    logging.info("Loading cohort information")
    for c in TCGA_COHORTS:
        logging.info(c)
        cc = load_clinical_cohort(c)
        ce = load_genexp_cohort(c)
        cc['cohort'] = c
        all_clinical.append(cc)
        all_genexp.append(ce)

    all_clinical = pd.concat(all_clinical, axis=0, sort=False)
    all_clinical['idx'] = np.arange(all_clinical.shape[0], dtype=int)
    all_genexp = np.vstack(all_genexp)
    
    logging.info("Loading stage patients from cbioPortal")
    stage_patients = pd.read_table(external_data_path / "cbioportal_tcga_stage.tsv")
    stage_patients = stage_patients.rename(columns={
        'Sample ID': 'sample', 
        'Neoplasm Disease Stage American Joint Committee on Cancer Code': 'ajcc_stage'
    })[['sample', 'ajcc_stage']]
    
    samples_with_stage = all_clinical.reset_index()[['sample', 'cohort', 'idx']].merge(stage_patients)
    
    def __add_stage_info(x):
        info = convert_stage_name(x['ajcc_stage'])
        y = x.copy()
        y['ajcc_stage_level'] = info[0]
        y['ajcc_stage_type'] = info[1]
        y['ajcc_stage_subtype'] = info[2]
        return y
    logging.info("Annotating the samples with stage information")
    samples_with_stage = samples_with_stage.apply(__add_stage_info, axis=1)
    samples_with_stage = samples_with_stage[samples_with_stage.ajcc_stage_level.notnull()]
    samples_with_stage['ajcc_stage_level'] = samples_with_stage['ajcc_stage_level'].astype(int)
    samples_with_stage['high_level_stage'] = samples_with_stage.ajcc_stage_level.map(lambda x: 'early' if x <= 2 else 'late')
    samples_with_stage = samples_with_stage.sort_values('idx')
    
    X = all_genexp[samples_with_stage.idx.values, :]
    samples_with_stage = samples_with_stage.drop('idx', axis=1)
    return X, samples_with_stage
    
    