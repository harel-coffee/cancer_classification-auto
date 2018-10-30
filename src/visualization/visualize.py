import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np


__usable_colors = [
    'blue',
    'red',
    'green'
]


def plot_PCA(X, y, labels, shape=(10, 7)):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    for yc in np.unique(y):
        X_pca_yc = X_pca[y == yc, :]
        sns.regplot(X_pca_yc[:, 0], X_pca_yc[:, 1], color=__usable_colors[yc], fit_reg=False, label=labels[yc])
    plt.legend()