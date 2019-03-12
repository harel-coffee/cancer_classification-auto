from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class TopVariantSelector(BaseEstimator, SelectorMixin):
    """ A very simple feature selector which uses the top variant features """

    def __init__(self, top_k):
        self.top_k = top_k

    def fit(self, X, y=None):
        stds = X.std(0)  # 1 x n.genes
        selected_genes = np.argsort(stds)[::-1][:self.top_k]
        self.selected_genes_ = selected_genes
        self.mask_ = np.in1d(np.arange(X.shape[1]), selected_genes)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_genes_')
        check_is_fitted(self, 'mask_')
        return self.mask_
