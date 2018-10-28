# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
from dotenv import find_dotenv, load_dotenv
from . import raw_data_path, processed_data_path, TCGA_COHORTS
from .genexp import load_raw_genexp_cohort
from .clinical import load_raw_clinical_cohort
import wget
import pandas as pd
import numpy as np


def download_TCGA_from_XenaBrowser():
    logger = logging.getLogger(__name__)
    genexp_url_template = "https://tcga.xenahubs.net/download/TCGA.{}.sampleMap/HiSeqV2.gz"
    clinical_url_template = "https://tcga.xenahubs.net/download/TCGA.{}.sampleMap/{}_clinicalMatrix.gz"
    
    out_path = raw_data_path / "TCGA" 
    os.makedirs(out_path, exist_ok=True)
    
    genexp_path = out_path / "genexp"
    os.makedirs(genexp_path, exist_ok=True)
    clinical_path = out_path / "clinical"
    os.makedirs(clinical_path, exist_ok=True)
    
    for cohort in TCGA_COHORTS:
        cohort_genexp_path = genexp_path / (cohort + ".gz")
        cohort_clinical_path = clinical_path / (cohort + ".gz")
        if not os.path.isfile(cohort_genexp_path):
            logger.info("Downloading gene expression of {}".format(cohort))
            wget.download(url=genexp_url_template.format(cohort), out = str(cohort_genexp_path))
        else:
            logger.info("Gene expression of {} already downloaded".format(cohort))
            
        if not os.path.isfile(cohort_clinical_path):
            logger.info("Downloading clinical data of {}".format(cohort))
            wget.download(url=clinical_url_template.format(cohort, cohort), out = str(cohort_clinical_path))
        else:
            logger.info("Clinical data of {} already downloaded".format(cohort))
           
            
def preprocess_TCGA_cohorts():
    logger = logging.getLogger(__name__)
    out_data_path = processed_data_path / "TCGA"
    os.makedirs(out_data_path, exist_ok=True)
    idx_to_gene, gene_to_idx = None, None
    
    all_clinical = []
    for cohort in TCGA_COHORTS:
        cohort_path = out_data_path / cohort
        os.makedirs(cohort_path, exist_ok=True)
        
        logger.info("Loading {} data".format(cohort))
        cohort_clinical = load_raw_clinical_cohort(cohort)
        cohort_genexp = load_raw_genexp_cohort(cohort)
        
        samples_intersection = sorted(set(cohort_clinical.index).intersection(set(cohort_genexp.index)))
        cohort_clinical = cohort_clinical.loc[samples_intersection]
        cohort_genexp = cohort_genexp.loc[samples_intersection]
        
        X = cohort_genexp.values
        logger.info("Creating {} data matrix".format(cohort))
        np.save(cohort_path / "X", X)
        
        logger.info("Creating {} clinical matrix".format(cohort))
        cohort_clinical.to_csv(cohort_path / "clinical.tsv", sep="\t", index=True)
        
        logger.info("Creating {} idx_to_sample.tsv".format(cohort))
        cohort_idx_to_sample = pd.Series(index=np.arange(cohort_clinical.shape[0], dtype=int), data = cohort_clinical.index)
        cohort_idx_to_sample.to_csv(cohort_path / "idx_to_sample.tsv", sep="\t", index=True, header=True)
        
        logger.info("Creating {} sample_to_idx.tsv".format(cohort))
        cohort_sample_to_idx = pd.Series(index = cohort_clinical.index, data = np.arange(cohort_clinical.shape[0], dtype=int))
        cohort_sample_to_idx.to_csv(cohort_path / "sample_to_idx.tsv", sep="\t", index=True, header=True)
        
        idx_to_gene_path = out_data_path / "idx_to_gene.tsv"
        if not os.path.isfile(idx_to_gene_path):
            logger.info("Creating {} idx_to_gene.tsv".format(cohort))
            idx_to_gene = pd.Series(index=np.arange(cohort_genexp.shape[1], dtype=int), data = cohort_genexp.columns)
            idx_to_gene.to_csv(idx_to_gene_path, sep="\t", index=True, header=True)
        else:
            logger.info("Checking idx_to_gene coherency with {}".format(cohort))
            cohort_idx_to_gene = pd.Series(index=np.arange(cohort_genexp.shape[1], dtype=int), data = cohort_genexp.columns)
            if not (cohort_idx_to_gene == idx_to_gene).all():
                raise ValueError("Error. idx_to_gene not consistent between datasets!!")
        
        gene_to_idx_path = out_data_path / "gene_to_idx.tsv"
        if not os.path.isfile(gene_to_idx_path):
            logger.info("Creating {} gene_to_idx.tsv".format(cohort))
            gene_to_idx = pd.Series(data=np.arange(cohort_genexp.shape[1], dtype=int), index=cohort_genexp.columns)
            gene_to_idx.to_csv(gene_to_idx_path, sep="\t", index=True, header=True)
        else:
            logger.info("Checking gene_to_idx coherency with {}".format(cohort))
            cohort_gene_to_idx = pd.Series(data=np.arange(cohort_genexp.shape[1], dtype=int), index = cohort_genexp.columns)
            if not (cohort_gene_to_idx == gene_to_idx).all():
                raise ValueError("Error. gene_to_idx not consistent between datasets!!")       
#         all_clinical.append(cohort_clinical)
    
#     all_clinical = pd.concat(all_clinical, axis=0, sort=True)
    
        
    
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading TCGA dataset from Xena Browser")
    download_TCGA_from_XenaBrowser()
    
    logger.info("Preprocessing TCGA cohorts")
    preprocess_TCGA_cohorts()
	

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
