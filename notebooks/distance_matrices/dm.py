import pandas as pd
import numpy as np
import sys
sys.path.append("../utils/")
import utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv1D, Flatten
from tensorflow import set_random_seed

import os
import random as rn

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
set_random_seed(seed)
rn.seed(seed)

# maximum number of cores
n_cores = 20

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=n_cores,
    inter_op_parallelism_threads=n_cores
)))


# ## Data

# In[ ]:


cancer_name = 'BRCA'
X_c, y_c = utils.get_cancer_data(cancer_name)
print("Cancer: {}".format(cancer_name))
print("\t#samples: {}".format(X_c.shape[0]))
print("\t#genes: {}".format(X_c.shape[1]))
print("\t#TUMORS: {}\t#NORMAL: {}".format(y_c[y_c == utils.TUMOR].shape[0], y_c[y_c == utils.NORMAL].shape[0]))


res_folder = "./results/ontological/{}/".format(cancer_name)
os.makedirs(res_folder, exist_ok=True)
out_path = os.path.join(res_folder, "results.xlsx")
print(out_path)
# ## Feature selection

# In[ ]:


def get_filtered_features(X):
    return np.arange(X.shape[1]) # nothing happens

def preprocess(X):
    scaler = MinMaxScaler()
    return utils.pre_process(X, get_filtered_features, scaler)


# ## Model creation

# In[ ]:


def create_conv_model(input_size):
    global stride
    print(stride)
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=(stride), input_shape=(input_size, 1), 
                     activation='relu', strides=stride))
    model.add(Flatten())
    model.add(Dense(units=200, activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid", name='output'))
    return model


# In[ ]:


def split_training_default_1(X, y, train, test, preprocess, validation_split, seed):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # get the validation set in a stratified fashion from the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split,
                                                      random_state=seed, stratify=y_train)
    print("Train - Test")
    # preprocess training set and get features and scaler
    X_train, scaler, sel_features = preprocess(X_train)
    print("Training scaled")
    # transform testing set
    X_test = scaler.fit_transform(X_test[:, sel_features])
    print("Test scaled")
    # transform validation set
    X_val = scaler.fit_transform(X_val[:, sel_features])
    print("Val scaled")
    oversampler = RandomOverSampler(random_state=seed)
    
    # oversampling
    X_train, y_train =oversampler.fit_sample(X_train, y_train)
    print("Train - oversampled")
    X_val, y_val = oversampler.fit_sample(X_val, y_val)
    print("Val - oversampled")
#     print(X_train.shape)
#     print(X_val.shape)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
    
    print("Reshaped")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ## Distance matrix

# In[ ]:


dm = np.load("/home/nanni/Data/TCGA/CIBB/ontological_distance_matrix.npy")
np.fill_diagonal(dm, np.inf)


# ### Neighbors

# In[ ]:


neighbors = np.argsort(dm, axis=1)


# ### Feature disposition

# In[ ]:


n_neighbors = 4
stride = n_neighbors + 1


# In[ ]:


conv_idxs = np.append(np.arange(neighbors.shape[0]).reshape(-1, 1), neighbors[:, :n_neighbors], axis=1).flatten()


# In[ ]:


X_c_conv = X_c[:, conv_idxs]


# ## Cross validation

# In[ ]:


cvscores_c, histories_c = utils.cross_validation(X=X_c_conv, y=y_c, 
                                                 preprocess=preprocess, 
                                                 seed=seed, 
                                                 data_preparation=split_training_default_1,
                                                 create_model=create_conv_model, 
                                                 get_measures=utils.get_measures)

# In[ ]:


cvscores_c.to_excel(out_path, index=False)

