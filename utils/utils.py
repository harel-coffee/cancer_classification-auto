from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import os

tumors_path = "/home/nanni/Data/TCGA/CIBB/"
all_tumor_names = list(filter(lambda x: os.path.isdir(os.path.join(tumors_path, x)) and
                              not x.startswith(".") and x != 'aggregates', os.listdir(tumors_path)))


def get_cancer_data(cancer_name):
    cancer_path = os.path.join(tumors_path, cancer_name)
    X = np.load(os.path.join(cancer_path, "X.npy"))
    y = np.load(os.path.join(cancer_path, "y.npy")).astype(int)
    return X, y


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
                     n_folds=5, n_epochs=100, batch_size=60,
                     validation_split=0.25, patience=10, metrics=['accuracy'],
                     optimizer="adam", loss='binary_crossentropy', seed=42,
                     data_preparation=split_training_default, **kwargs):
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    early_stopping = get_early_stopping_condition(patience=patience)

    cvscores = []
    histories = []
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
        print("".join(["{:<10}{:<10.2f}".format(k, v) for (k, v) in measures.items()]))
        cvscores.append(measures)

    cvscores = pd.DataFrame.from_dict(cvscores)
    return cvscores, histories


def report(cvscores, writer, sheet_name=None):
    means = cvscores.mean()
    means.name = "mean"
    r = pd.concat((cvscores.set_index("split"), means.to_frame().T.drop("split", axis=1)))
    r.to_excel(writer, sheet_name=sheet_name)