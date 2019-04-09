import os
from .. import project_dir

# ---------- data locations ----------
data_path = project_dir / "data"

# raw data
raw_data_path = data_path / "raw"
raw_TCGA_path = raw_data_path / "TCGA"
# processed
processed_data_path = data_path / "processed"
processed_TCGA_path = processed_data_path / "TCGA"
# interim
interim_data_path = data_path / "interim"
external_data_path = data_path / "external"

TCGA_COHORTS = [
	'LAML',
	'ACC',
	'CHOL',
	'BLCA',
	'BRCA',
	'CESC',
	'COAD',
	'UCEC',
	'ESCA',
	'GBM',
	'HNSC',
	'KICH',
	'KIRC',
	'KIRP',
	'DLBC',
	'LIHC',
	'LGG',
	'LUAD',
# 	'LUNG',
	'LUSC',
	'SKCM',
	'MESO',
	'UVM',
	'OV',
	'PAAD',
	'PCPG',
	'PRAD',
	'READ',
	'SARC',
	'STAD',
	'TGCT',
	'THYM',
	'THCA',
	'UCS'
]

from .genexp import *
from .clinical import *
from .classification_problems import *