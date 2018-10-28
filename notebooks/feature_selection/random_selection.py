import sys
import os
import random as rn
sys.path.append("../utils/")
import utils

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow import set_random_seed

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
set_random_seed(seed)
rn.seed(seed)

# maximum number of cores
n_cores = 10

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=n_cores,
    inter_op_parallelism_threads=n_cores
)))


cancer_name = 'BRCA'
X_c, y_c = utils.get_cancer_data(cancer_name)
print("Cancer: {}".format(cancer_name))
print("\t#samples: {}".format(X_c.shape[0]))
print("\t#genes: {}".format(X_c.shape[1]))
print("\t#TUMORS: {}\t#NORMAL: {}".format(y_c[y_c == utils.TUMOR].shape[0], y_c[y_c == utils.NORMAL].shape[0]))

def get_random_genes(k, n):
    return np.random.choice(range(n), k, replace=False)


k = 400
n = X_c.shape[1]

times = 1000

def get_filtered_features(X):
    return np.arange(X.shape[1])

def preprocess(X):
    scaler = MinMaxScaler()
    return utils.pre_process(X, get_filtered_features, scaler)


def net(input_size):
    """ A super-simple NN for the single tumor classification
    """
    model = Sequential()
    model.add(Dense(100, input_shape=(input_size,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    return model


cvscoress_all = None
out_file = "./results/random_selection.tsv"

for i in range(times):
    rg = get_random_genes(k, n)
    print("Random {} genes".format(len(rg)))
    X_c_i = X_c[:, rg]
    cvscores_c_i, histories_c_i = utils.cross_validation(X=X_c_i, y=y_c,
                                                         preprocess=preprocess,
                                                         seed=seed,
                                                         create_model=net,
                                                         get_measures=utils.get_measures)
    cvscores_c_i['experiment'] = i
    cvscoress_all = pd.concat([cvscoress_all, cvscores_c_i], axis=0)
    print("Saving file {}".format(out_file))
    cvscoress_all.to_csv(out_file, sep="\t", index=False, header=True)