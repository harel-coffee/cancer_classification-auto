from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from MulticoreTSNE import MulticoreTSNE as TSNE

from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt

TUMOR = 0
NORMAL = 1

tb_logs = "/home/nanni/tensorboard_logs"
tb_session_name = "SAE_KIRC"

tumors_path = "/home/nanni/Data/TCGA/CIBB/"
aggregates_path = "/home/nanni/Data/TCGA/CIBB/aggregates"
all_tumor_names = list(filter(lambda x: os.path.isdir(os.path.join(tumors_path, x)) and
                              not x.startswith(".") and x != 'aggregates', os.listdir(tumors_path)))

all_aggregate_names = list(filter(lambda x: os.path.isdir(os.path.join(aggregates_path, x)) and not x.startswith("."),
                                  os.listdir(aggregates_path)))


def set_tb_logs(path):
    global tb_logs
    tb_logs = path

    
def set_tb_session_name(name):
    global tb_session_name
    tb_session_name = name


def get_tensorboard_callback(session_name=None):
    if session_name is None:
        session_name = tb_session_name 
    return TensorBoard(log_dir="{}/{}__{}".format(tb_logs, session_name, time.strftime('%Y_%m_%d__%H_%M')))


def get_cancer_data(cancer_name):
    if cancer_name in all_tumor_names:
        cancer_path = os.path.join(tumors_path, cancer_name)
    elif cancer_name in all_aggregate_names:
        cancer_path = os.path.join(aggregates_path, cancer_name)
    else:
        raise ValueError("{} does not exist!".format(cancer_name))
    X = np.load(os.path.join(cancer_path, "X.npy"))
    y = np.load(os.path.join(cancer_path, "y.npy")).astype(int)
    
    # print a summary
    print("Cancer: {}".format(cancer_name))
    print("\t#samples: {}".format(X.shape[0]))
    print("\t#genes: {}".format(X.shape[1]))
    print("\t#TUMORS: {}\t#NORMAL: {}".format(y[y == TUMOR].shape[0], y[y == NORMAL].shape[0]))

    return X, y


def get_all_tcga(data="all"):
    all_tcga = None
    labels = []
    for cname in all_tumor_names:
        X_c, y = get_cancer_data(cname)
        if all_tcga is None:
            n_genes = X_c.shape[1]
            all_tcga = np.empty((0, n_genes))
        if data == 'tumor':
            X_c = X_c[y == TUMOR, :]
        elif data == 'normal':
            X_c = X_c[y == NORMAL, :]
        elif data == 'all':
            pass
        else:
            raise ValueError("data filed is incorrect")
        all_tcga = np.vstack((all_tcga, X_c))
        labels.extend([cname for _ in range(X_c.shape[0])])
    return all_tcga, labels


def get_measures(y_true, y_pred):
    f1 = f1_score(y_pred=y_pred, y_true=y_true)
    precision = precision_score(y_pred=y_pred, y_true=y_true)
    recall = recall_score(y_pred=y_pred, y_true=y_true)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

    return {'f1-score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy}


def pre_process(X, get_filtered_features, scaler):
    sel_features = get_filtered_features(X)
    X_filtered = X[:, sel_features]

    scaler.fit(X_filtered)
    X_transf = scaler.transform(X_filtered)
    return X_transf, scaler, sel_features


def get_early_stopping_condition(monitor="val_loss",
                                 min_delta=0,
                                 patience=0,
                                 mode='auto'):
    return EarlyStopping(monitor=monitor,
                         min_delta=min_delta,
                         patience=patience,
                         verbose=0,
                         mode=mode)


def split_training_default(X, y, train, test, preprocess, validation_split, seed):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # get the validation set in a stratified fashion from the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split,
                                                      random_state=seed, stratify=y_train)

    # preprocess training set and get features and scaler
    X_train, scaler, sel_features = preprocess(X_train)

    # transform testing set
    X_test = scaler.fit_transform(X_test[:, sel_features])

    # transform validation set
    X_val = scaler.fit_transform(X_val[:, sel_features])
    return X_train, X_val, X_test, y_train, y_val, y_test


def cross_validation(X, y, create_model, preprocess=None, get_measures=get_measures,
                     n_folds=5, n_epochs=100, batch_size=60, n_randomizations=1,
                     validation_split=0.25, patience=10, metrics=['accuracy'],
                     optimizer="adam", loss='binary_crossentropy', seed=42,
                     data_preparation=split_training_default, **kwargs):
    early_stopping = get_early_stopping_condition(patience=patience)

    cvscores = []
    histories = []

    for r in range(n_randomizations):
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for i_split, (train, test) in enumerate(kfold.split(X, y)):
            X_train, X_val, X_test, y_train, y_val, y_test = data_preparation(X, y, train, test, preprocess,
                                                                              validation_split, seed, **kwargs)

            model = create_model(X_train.shape[1])
            # compile the model
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            # fit the model
            history = model.fit(X_train, y_train,
                                epochs=n_epochs, batch_size=batch_size,
                                verbose=0, validation_data=(X_val, y_val),
                                callbacks=[early_stopping])
            histories.append(history)
            # evaluate the model
            y_pred = model.predict_classes(X_test)
            measures = get_measures(y_pred=y_pred, y_true=y_test)
            measures['split'] = i_split
            measures['random_set'] = r
            print("".join(["{:<10}{:<10.2f}".format(k, v) for (k, v) in measures.items()]))
            cvscores.append(measures)

    cvscores = pd.DataFrame.from_dict(cvscores)
    return cvscores, histories


def report(cvscores, writer, sheet_name=None):
    means = cvscores.mean()
    means.name = "mean"
    r = pd.concat((cvscores.set_index("split"), means.to_frame().T.drop("split", axis=1)))
    r.to_excel(writer, sheet_name=sheet_name)


""" Staging Utils"""

# Definitions
STAGE_COLUMN = "pathologic_stage"
SAMPLE_TYPE_COLUMN = "sample_type"

STAGE_I = "Stage I"
STAGE_II = "Stage II"
STAGE_III = "Stage III"
STAGE_IV = "Stage IV"
STAGES = [STAGE_I, STAGE_II, STAGE_III, STAGE_IV]

EARLY_STAGE = "Early_Stage"
LATE_STAGE = "Late_Stage"

PRIMARY_TUMOR_TYPE = "Primary Tumor"
SOLID_TISSUE_NORMAL = "Solid Tissue Normal"

class_encoding = {
    EARLY_STAGE: 0,
    LATE_STAGE: 1
}

class_decoding = {v: k for k, v in class_encoding.items()}

GRADE_COLUMN = "neoplasm_histologic_grade"
GRADE_1 = "G1"
GRADE_2 = "G2"
GRADE_3 = "G3"
GRADE_4 = "G4"
GRADE_UNKNOWN = "GX"
GRADES = [GRADE_1, GRADE_2, GRADE_3, GRADE_4]

LOW_GRADE = "Low Grade"
HIGH_GRADE = "High Grade"

grade_encoding = {
    LOW_GRADE: 0,
    HIGH_GRADE: 1
}

grade_decoding = {v: k for k, v in grade_encoding.items()}


def stage_to_class(stage):
    """Mapping the stage to the classes"""
    if stage in [STAGE_I, STAGE_II]:
        return class_encoding[EARLY_STAGE]
    else:
        return class_encoding[LATE_STAGE]


def encode_class(c):
    return class_encoding[c]


def decode_class(c):
    return class_decoding[c]


def __load_cohort_data(X_path, X_meta_path):
    X_exp = pd.read_csv(X_path, sep="\t").set_index("sample").T
    X_meta = pd.read_csv(X_meta_path, sep="\t").set_index("sampleID")
    return X_exp, X_meta


def load_classification_problem(X_path, X_meta_path, 
                                y_column, y_subset, 
                                X_sample_type, label_to_class=lambda x: x):
    X_exp, X_meta = __load_cohort_data(X_path, X_meta_path)
    patients = X_meta[X_meta[y_column].isin(y_subset) & \
                      (X_meta[SAMPLE_TYPE_COLUMN] == X_sample_type)].index \
                     .intersection(X_exp.index).tolist()
    print("# patients with {} and {}: {}".format(X_sample_type, ", ".join(y_subset), len(patients)))
    
    X_meta_s = X_meta.loc[patients]
    gene_names = X_exp.columns.tolist()
    X = X_exp.loc[patients].as_matrix(gene_names)
    y = np.array(X_meta_s[y_column].map(label_to_class).tolist())

    assert X.shape[0] == y.shape[0], "X and y have different number of samples"

    idx_to_patient = pd.Series(index=np.arange(len(patients)), data=patients)
    patient_to_idx = pd.Series(data=np.arange(len(patients)), index=patients)

    idx_to_gene = pd.Series(index=np.arange(len(gene_names)), data=gene_names)
    gene_to_idx = pd.Series(data=np.arange(len(gene_names)), index=gene_names)

    return X, y, idx_to_patient, patient_to_idx, idx_to_gene, gene_to_idx
    

def grade_to_class(grade):
    if grade in [GRADE_1, GRADE_2]:
        return grade_encoding[LOW_GRADE]
    else:
        return grade_encoding[HIGH_GRADE]


def load_grade_data(X_path, X_meta_path):
    return load_classification_problem(X_path, X_meta_path, 
                                       GRADE_COLUMN, GRADES, 
                                       PRIMARY_TUMOR_TYPE,
                                       grade_to_class)


def load_stage_data(X_path, X_meta_path):
    X_exp, X_meta = __load_cohort_data(X_path, X_meta_path)

    patients = X_meta[X_meta[STAGE_COLUMN].isin(STAGES) & \
                      (X_meta[SAMPLE_TYPE_COLUMN] == PRIMARY_TUMOR_TYPE)].index \
        .intersection(X_exp.index).tolist()

    # normal_samples = X_meta[(X_meta[SAMPLE_TYPE_COLUMN] == SOLID_TISSUE_NORMAL)].index \
    #     .intersection(X_exp.index).tolist()

    print("# patients with {} and {}: {}".format(PRIMARY_TUMOR_TYPE, ", ".join(STAGES), len(patients)))

    X_meta_s = X_meta.loc[patients]
    gene_names = X_exp.columns.tolist()
    X = X_exp.loc[patients].as_matrix(gene_names)
    #     X_n = X_exp.loc[normal_samples].as_matrix()
    y = np.array(X_meta_s[STAGE_COLUMN].map(stage_to_class).tolist())
    #     y_n = [class_encoding[NORMAL] for _ in range(X_n.shape[0])]
    #     X_with_normal = np.append(X, X_n, axis=0)
    #     y_with_normal = np.append(y, y_n)

    assert X.shape[0] == y.shape[0], "X and y have different number of samples"

    idx_to_patient = pd.Series(index=np.arange(len(patients)), data=patients)
    patient_to_idx = pd.Series(data=np.arange(len(patients)), index=patients)

    idx_to_gene = pd.Series(index=np.arange(len(gene_names)), data=gene_names)
    gene_to_idx = pd.Series(data=np.arange(len(gene_names)), index=gene_names)

    return X, y, idx_to_patient, patient_to_idx, idx_to_gene, gene_to_idx


""" Stage scoring """


name_labels = [EARLY_STAGE, LATE_STAGE]
labels = list(map(lambda x: class_encoding[x], name_labels))


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels)[0, 0]


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels)[1, 1]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels)[1, 0]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels)[0, 1]


"""  Plotting utils """


def plot_PCA(X, y):
    pca = PCA(n_components=2)
    X_t = pca.fit_transform(X)
    plt.scatter(X_t[:, 0], X_t[:, 1], c=LabelEncoder().fit_transform(y))


def plot_TSNE(X, y, n_jobs=4):
    tsne = TSNE(n_components=2, n_jobs=n_jobs)
    X_t = tsne.fit_transform(X)
    plt.scatter(X_t[:, 0], X_t[:, 1], c=LabelEncoder().fit_transform(y))
    
    
"""
    Loading all TCGA
"""

def load_PANCAN_TCGA(path="./data/PANCAN_raw/", 
                     geneexp_name="batchRemoved", 
                     clinical_name="clinicalMatrix",
                     sample_types_name="sample_type",
                     cnv_name="CNV"):
    print("LOADING GENE EXPRESSION")
    tcga = load_geneexp_data(os.path.join(path, "PANCAN_{}.tsv".format(geneexp_name)))
    print("LOADING CLINICAL DATA")
    clinical = load_clinical_data(os.path.join(path, "PANCAN_{}.tsv".format(clinical_name)))
    print("LOADING SAMPLE INFORMATION")
    sample_types = load_sample_types(os.path.join(path, "PANCAN_{}.tsv".format(sample_types_name)))
    print("LOADING CNV DATA")
    cnv = load_cnv_data(os.path.join(path, "PANCAN_{}.tsv".format(cnv_name)))
    
    tcga_samples = set(tcga.index.tolist())
    clinical_samples = set(clinical.index.tolist())
    sample_types_samples = set(sample_types.index.tolist())
    cnv_samples = set(cnv.index.tolist())
    
    tcga_genes = set(tcga.columns.tolist())
    cnv_genes = set(cnv.columns.tolist())
    
    genes_intersection = sorted(tcga_genes.intersection(cnv_genes))
    
    sample_intersection = list(tcga_samples.intersection(clinical_samples)\
                               .intersection(sample_types_samples)\
                               .intersection(cnv_samples))
    
    tcga = tcga.loc[tcga.index.intersection(sample_intersection), genes_intersection].sort_index()
    clinical = clinical.loc[clinical.index.intersection(sample_intersection)].sort_index()
    sample_types = sample_types.loc[sample_types.index.intersection(sample_intersection)].sort_index()
    cnv = cnv.loc[cnv.index.intersection(sample_intersection), genes_intersection].sort_index()
    return tcga, cnv, clinical, sample_types
    

def load_geneexp_data(path="./data/PANCAN_raw/PANCAN_batchRemoved.tsv"):
    print("Loading expression data")
    tcga = pd.read_csv(path, sep="\t")
    # remove the NaN values due to batch-normalization
    print("Removing NaN values (filling with zero)")
    tcga = tcga.fillna(0)
    # there is a gene which has multiple occurrencies. 
    # We deal with it by doing the mean of counts.
    print("Grouping duplicated genes by mean")
    tcga = tcga.groupby(by="sample").mean()
    # give a name to the gene index
    tcga.index.rename("gene_symbol", inplace=True)
    # give a name to the patient index
    tcga.columns.rename("sample", inplace=True)
    # convert to patient x gene matrix
    print("Transpose and sort the data")
    tcga = tcga.T
    # sort the patient index
    tcga.sort_index(axis=0, inplace=True)
    # sort the gene index
    tcga.sort_index(axis=1, inplace=True)
    return tcga


def load_cnv_data(path="./data/PANCAN_raw/PANCAN_CNV.tsv"):
    cnv = pd.read_csv(path, sep="\t")
    cnv = cnv.rename(columns={'Sample': 'sample'}).set_index("sample").T
    cnv.columns.rename("gene_symbol", inplace=True)
    return cnv


def load_clinical_data(path="./data/PANCAN_raw/PANCAN_clinicalMatrix.tsv"):
    clinical = pd.read_csv(path, sep="\t")
    clinical.set_index("sample", inplace=True)
    return clinical


def load_sample_types(path="./data/PANCAN_raw/PANCAN_sample_type.tsv"):
    sample_types = pd.read_csv(path, sep="\t")
    sample_types.set_index("sample", inplace=True)
    return sample_types


def to_matrix(tcga):
    idx_to_sample = pd.Series(data=tcga.index.tolist(), index=np.arange(tcga.shape[0], dtype=int))
    idx_to_gene = pd.Series(data=tcga.columns.tolist(), index=np.arange(tcga.shape[1], dtype=int))
    m = tcga.as_matrix()
    return m, idx_to_sample, idx_to_gene


def save_dataset_to_matrix(tcga, clinical, sample_types,
                           path="./data/PANCAN_preprocessed/"):
    os.makedirs(path, exist_ok=True)
    # save gene expression
    m, idx_to_sample, idx_to_gene = to_matrix(tcga)
    np.save(os.path.join(path, "data_preprocessed"), m)
    idx_to_sample.to_csv(os.path.join(path, "idx_to_sample.csv"), index=True, header=False)
    idx_to_gene.to_csv(os.path.join(path, "idx_to_gene.csv"), index=True, header=False)
    
    # save clinical
    clinical.to_csv(os.path.join(path, "clinical_preprocessed.csv"), index=True, header=True)
    # save sample_types
    sample_types.to_csv(os.path.join(path, "sample_types_preprocessed.csv"), index=True, header=True)

    
def load_PANCAN_TCGA_from_matrix(path="./data/PANCAN_preprocessed/"):
    m = np.load(os.path.join(path, "data_preprocessed.npy"))
    idx_to_sample = pd.read_csv(os.path.join(path, "idx_to_sample.csv"), 
                                index_col=0, header=None, squeeze=True)
    idx_to_gene = pd.read_csv(os.path.join(path, "idx_to_gene.csv"), 
                              index_col=0, header=None, squeeze=True)
    
    clinical = pd.read_csv(os.path.join(path, "clinical_preprocessed.csv"), 
                           index_col=0)
    sample_types = pd.read_csv(os.path.join(path, "sample_types_preprocessed.csv"), 
                               index_col=0)
    
    return m, idx_to_sample, idx_to_gene, clinical, sample_types