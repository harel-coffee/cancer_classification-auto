import pandas as pd


class Dataset(object):
    """Class embedding a generic dataset"""
    
    def __init__(self, X, 
                 idx_to_sample=None, sample_to_idx=None,
                 idx_to_column=None, column_to_idx=None,
                 y=None, y_mapping=None):
        self.__X = X
        
        if ((idx_to_sample is None) and (sample_to_idx is None)) or \
            ((idx_to_sample is not None) and (sample_to_idx is not None)):
            self.__idx_to_sample = idx_to_sample
            self.__sample_to_idx = sample_to_idx
        elif (idx_to_sample is not None) and (sample_to_idx is None):
            self.__sample_to_idx = pd.Series(index=idx_to_sample.values, data=idx_to_sample.index)
        else:
            self.__idx_to_sample = pd.Series(index=sample_to_idx.values, data=sample_to_idx.index)
            
        if ((idx_to_column is None) and (column_to_idx is None)) or \
            ((idx_to_column is not None) and (column_to_idx is not None)):
            self.__idx_to_column = idx_to_column
            self.__column_to_idx = column_to_idx
        elif (idx_to_column is not None) and (column_to_idx is None):
            self.__column_to_idx = pd.Series(index=idx_to_column.values, data=idx_to_column.index)
        else:
            self.__idx_to_column = pd.Series(index=column_to_idx.values, data=column_to_idx.index)
        
        self.__y = y
        self.__y_mapping = y_mapping
    
    @property
    def data(self):
        return self.__X
    
    @property
    def idx_to_sample(self):
        return self.__idx_to_sample
    
    @property
    def sample_to_idx(self):
        return self.__sample_to_idx
    
    @property
    def y(self):
        return self.__y
    
    @property
    def y_names(self):
        return self.__y_mapping
    
    @property
    def idx_to_column(self):
        return self.__idx_to_column
    
    @property
    def column_to_idx(self):
        return self.__column_to_idx