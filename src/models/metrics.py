import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate

def __cm_0_0(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]
def __cm_0_1(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]
def __cm_1_0(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]
def __cm_1_1(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def get_confusion_matrix_scores(y):
    classes = sorted(np.unique(y))
    scorers = {}
    
    if(len(classes) == 2):
        scorers["0_0"] = make_scorer(__cm_0_0)
        scorers["0_1"] = make_scorer(__cm_0_1)
        scorers["1_0"] = make_scorer(__cm_1_0)
        scorers["1_1"] = make_scorer(__cm_1_1)
    else:
        raise ValueError("Error! For now handling only two classes metrics!")
    return scorers

def to_confusion_matrices(cv_scores):
    coords = list(map(lambda y: (int(y.split("_")[1]), int(y.split("_")[2])),
                  filter(lambda x: "test_" in x, cv_scores.keys())))
    n_classes = len(set(map(lambda x: x[0], coords)))
    n_folds = cv_scores['fit_time'].shape[0]
    X = np.zeros((n_folds, n_classes, n_classes), dtype=int)
    for i, j in coords:
        key_ij = "test_{}_{}".format(i, j)
        value_ij = cv_scores[key_ij]
        X[:, i, j] = value_ij
    return X


def cross_validation(pipeline, X, y=None, scoring=None, cv=5, n_jobs=1):
    if scoring is None:
        scoring = get_confusion_matrix_scores(y)
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)
    return to_confusion_matrices(scores)