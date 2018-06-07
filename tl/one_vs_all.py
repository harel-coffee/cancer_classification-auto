import time
import pandas as pd
import numpy as np
import random as rn
from tqdm import tqdm
import os

import sys
sys.path.append("../utils/")
import utils

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

TUMOR = 0
NORMAL = 1


def get_result_folder_name(cancer_name):
    now = time.strftime('%Y%m%d_%H%M')
    description = "{}_1vsALL_NN_NN".format(cancer_name)
    folder = now + "_" + description
    output_folder = os.path.join("./results/", folder)
    return output_folder


def get_filtered_features(X):
    # return np.arange(X.shape[1])
    return X.std(0).argsort()[::-1][:5000]


def preprocess(X):
    scaler = MinMaxScaler()
    return utils.pre_process(X, get_filtered_features, scaler)


def tumor_alone_model(input_size):
    """ A super-simple NN for the single tumor classification
    """
    model = Sequential()
    model.add(Dense(100, input_shape=(input_size,), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    return model


def others_alone_model(input_size):
    h1 = 500
    h2 = 200
    h3 = 100
    h4 = 50
    out = 1

    model = Sequential()
    model.add(Dense(h1, input_shape=(input_size,), activation="relu"))
    model.add(Dense(h2, activation="relu"))
    model.add(Dense(h3, activation="relu"))
    model.add(Dense(h4, activation="relu"))
    model.add(Dense(out, activation="sigmoid"))
    return model


def create_other_network(input_size):
    h1 = 500
    h2 = 200
    h3 = 100
    h4 = 50
    out = 1

    model = Sequential()
    model.add(Dense(h1, input_shape=(input_size,), activation="relu", name='h1'))
    model.add(Dense(h2, activation="relu", name='h2'))
    model.add(Dense(h3, activation="relu", name='h3'))
    model.add(Dense(h4, activation="relu", name='h4'))
    model.add(Dense(out, activation="sigmoid", name='out'))

    encoder = Model(inputs=model.input, outputs=model.get_layer("h3").output)

    return model, encoder


def create_additional_network(input_size):
    h1 = 50
    h2 = 10
    out = 1

    model = Sequential()
    model.add(Dense(h1, input_shape=(input_size,), activation="relu", name='h1'))
    model.add(Dense(h2, activation="relu", name='h2'))
    model.add(Dense(out, activation="sigmoid", name='out'))
    return model


def transfer_learning(X, y, train, test, preprocess, validation_split, seed, X_other, y_other):
    #     print(X.shape, y.shape)
    #     print(X_other.shape, y.shape)
    #     print("Splitting of X_c")
    # Splitting the single tumor dataset
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # get the validation set in a stratified fashion from the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split,
                                                      random_state=seed, stratify=y_train)
    #     print("Scaling of X_c")
    # preprocess training set and get features and scaler
    X_train, scaler, sel_features = preprocess(X_train)
    # transform testing set
    X_test = scaler.fit_transform(X_test[:, sel_features])
    # transform validation set
    X_val = scaler.fit_transform(X_val[:, sel_features])

    #     print("Scaling and selection on X_other")
    # for the other set we use a brand new scaler but the same features
    other_scaler = MinMaxScaler()
    X_other = other_scaler.fit_transform(X_other[:, sel_features])
    # splitting other set in training and validation (no test...useless)
    X_other_train, X_other_val, y_other_train, y_other_val = train_test_split(X_other, y_other,
                                                                              test_size=validation_split,
                                                                              random_state=seed, stratify=y_other)

    #     print("Fitting the OTHER model")
    # create and fit the OTHER model
    other_model, encoder = create_other_network(input_size=X_other_train.shape[1])
    other_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    other_model.fit(X_other_train, y_other_train,
                    epochs=100, batch_size=60,
                    verbose=0, validation_data=(X_other_val, y_other_val),
                    callbacks=[utils.get_early_stopping_condition()])

    #     print("Encoding X_c")
    # embedding of data
    X_train_code = encoder.predict(X_train)
    X_val_code = encoder.predict(X_val)
    X_test_code = encoder.predict(X_test)

    #     print(X_train_code.shape)
    #     print(X_val_code.shape)
    #     print(X_test_code.shape)

    return X_train_code, X_val_code, X_test_code, y_train, y_val, y_test


if __name__ == '__main__':
    all_tumors = utils.all_tumor_names
    for cancer_name in all_tumors:
        print(cancer_name)
        c_result_folder = get_result_folder_name(cancer_name)
        os.makedirs(c_result_folder, exist_ok=True)
        writer = pd.ExcelWriter(os.path.join(c_result_folder, "results.xlsx"), engine='xlsxwriter')

        # selected cancer
        X_c, y_c = utils.get_cancer_data(cancer_name)
        print("\t#samples: {}".format(X_c.shape[0]))
        print("\t#genes: {}".format(X_c.shape[1]))
        n_tumors, n_normals = y_c[y_c == TUMOR].shape[0], y_c[y_c == NORMAL].shape[0]
        print("\t#TUMORS: {}\t#NORMAL: {}".format(n_tumors, n_normals))
        if n_normals < 5:
            print("\t\tToo low number of normal samples --> REJECTED")
            continue

        # all the others
        others = list(set(utils.all_tumor_names) - {cancer_name})
        X_others = np.empty((0, X_c.shape[1]), dtype=int)
        y_others = np.empty(0, dtype=int)

        for o in others:
            # print(o)
            X_o, y_o = utils.get_cancer_data(o)
            X_others = np.append(X_others, X_o, axis=0)
            y_others = np.append(y_others, y_o)

        # Test on cancer ALONE
        print("\t {} ALONE".format(cancer_name))
        cvscores_c, histories_c = utils.cross_validation(X=X_c, y=y_c, preprocess=preprocess, seed=seed,
                                                         create_model=tumor_alone_model,
                                                         get_measures=utils.get_measures)
        utils.report(cvscores_c, writer=writer, sheet_name="{}_alone".format(cancer_name))

        # Test on others ALONE
        print("\t {} ALONE".format("ALL"))
        cvscores_others, histories_others = utils.cross_validation(X=X_others, y=y_others, preprocess=preprocess,
                                                                   seed=seed, create_model=others_alone_model,
                                                                   get_measures=utils.get_measures)
        utils.report(cvscores_others, writer=writer, sheet_name="{}_others".format(cancer_name))

        # Test on TRANSFER LEARNING
        print("\t {} TRANSFER LEARNING WITH ALL".format(cancer_name))
        cvscores_tl, histories_tl = utils.cross_validation(X=X_c, y=y_c, preprocess=preprocess, seed=seed,
                                                           create_model=create_additional_network,
                                                           get_measures=utils.get_measures,
                                                           data_preparation=transfer_learning,
                                                           X_other=X_others, y_other=y_others)
        utils.report(cvscores_tl, writer=writer, sheet_name="{}_TL".format(cancer_name))

        writer.save()
