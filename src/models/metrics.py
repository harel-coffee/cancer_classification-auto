import pandas as pd


def read_confusion_matrix(path):
    cm = pd.read_table(path, index_col=0)
    cm.index.name = "Observed"
    cm.columns.name = "Predicted"
    return cm