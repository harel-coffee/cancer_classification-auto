import numpy as np
import sys
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate

# dynamically define the confusion matrix accession functions

__max_cm_size = 10
__this_module = sys.modules[__name__]

for i in range(__max_cm_size):
    for j in range(__max_cm_size):
        exec("def __cm_{}_{}(y_true, y_pred):\n\treturn confusion_matrix(y_true, y_pred)[{}, {}]".format(i, j, i, j))


def get_confusion_matrix_scores(y):
    classes = sorted(np.unique(y))
    scorers = {}
    for i in classes:
        for j in classes:
            scorers["{}_{}".format(i, j)] = make_scorer(getattr(__this_module, "__cm_{}_{}".format(i, j)))
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
        f_scoring = get_confusion_matrix_scores(y)
    else:
        f_scoring = scoring
    scores = cross_validate(pipeline, X, y, scoring=f_scoring, cv=cv, n_jobs=n_jobs)
    
    if scoring is None:
        return to_confusion_matrices(scores)
    else:
        return scores