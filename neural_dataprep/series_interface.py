"""
Used inside FeatureTransformers to make dataframe agnostic transformations
"""
import pandas as pd
import polars as pl
import numpy as np
from abc import ABC, abstractmethod

class SeriesInterface(ABC):
    """ Meant as an interface for series of types `polars`, pandas, and `numpy` """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def copy(self, series):
        return series
    
    @abstractmethod
    def fillna(self, series, value=None):
        return series
    
    @abstractmethod
    def mean(self,series):
        return series
    
    @abstractmethod
    def max(self, series):
        return series

    @abstractmethod
    def min(self, series):
        return series
    
    @abstractmethod
    def std(self,series):
        return series

    @abstractmethod
    def nunique(self,series):
        return series
    
    @abstractmethod
    def unique(self,series):
        return series
    
class PandasInterface(SeriesInterface):
    """ Interface for pandas Series """

    def __init__(self):
        super().__init__()

    def copy(self, series):
        return series.copy()
    
    def fillna(self, series, value=None):
        return series.fillna(value)

    def mean(self, series):
        return series.mean()
    
    def map_dict(self, series, dict_):
        if series.dtype.name == 'object':
            return series.fillna('null_value_999').map(dict_).fillna(len(dict_.keys()))
        else:
            return series.fillna(-999).map(dict_).fillna(len(dict_.keys()))

    def max(self, series):
        return series.max(skipna=True)

    def min(self, series):
        return series.min(skipna=True)

    def std(self, series):
        return series.std()

    def nunique(self, series):
        return series.nunique()

    def unique(self, series):
        return series.unique()
    
    def value_counts(self, series):
        if series.dtype.name == 'object':
            return pd.DataFrame(series.fillna('null_value_999').value_counts()).reset_index()
        else:
            return pd.DataFrame(sseries.fillna(-999).value_counts()).reset_index()
    
class PolarsInterface(SeriesInterface):
    """ Implementation of SeriesInterface for polars Series """

    def __init__(self):
        super().__init__()

    def copy(self, series):
        return series.clone()
    
    def map_dict(self, series, dict_):
        def map_func(x):
            return dict_.get(x, x)  # Returns the mapped value, or the original if not in dict

        if series.dtype == pl.datatypes.String:
            return series.fill_null('null_value_999').map_elements(map_func, return_dtype=pl.Int64).fill_null(len(dict_.keys()))
        else:
            return series.fill_null(-999).map_elements(map_func, return_dtype=pl.Int64).fill_null(len(dict_.keys()))
    
    def max(self, series):
        return series.fill_nan(None).max()

    def min(self, series):
        return series.fill_nan(None).min()

    def mean(self, series):
        return series.fill_nan(None).mean()

    def std(self, series):
        return series.fill_nan(None).std(ddof=1)

    def nunique(self, series):
        return series.fill_nan(None).n_unique()

    def unique(self, series):
        return series.fill_nan(None).unique()

    def fillna(self, series, value=None):
        return series.fill_nan(value)
    
    def value_counts(self, series):
        if series.dtype == pl.datatypes.String:
            return series.fill_null('null_value_999').value_counts().to_pandas()
        else:
            return series.fill_nan(-999).value_counts().to_pandas()
    
class NumpyInterface(SeriesInterface):
    """ Implementation of SeriesInterface for numpy arrays """

    def __init__(self):
        super().__init__()

    def copy(self, array):
        return np.copy(array)

    def mean(self, array):
        return np.nanmean(array)

    def max(self, array):
        return np.nanmax(array)

    def min(self, array):
        return np.nanmin(array)

    def std(self, array):
        return np.nanstd(array, ddof=1)

    def nunique(self, array):
        return np.unique(array[~np.isnan(array)]).size

    def unique(self, array):
        return np.unique(array[~np.isnan(array)])

    def fillna(self, array, value=None):
        filled_array = np.copy(array)
        np.nan_to_num(filled_array, copy=False, nan=value)
        return filled_array

class SeriesFactory:
    @staticmethod
    def get_interface(series):
        if isinstance(series, pd.Series):
            return PandasInterface()
        elif isinstance(series, pl.Series):
            return PolarsInterface()
        elif isinstance(series, np.ndarray):
            return NumpyInterface()
        else:
            raise ValueError("Unsupported series type. Must be pandas Series, polars Series, or numpy array.")
####################################################################################################################
#Helper functions
####################################################################################################################
def get_series_interface(series):
    return SeriesFactory.get_interface(series)

def copy_dataframe(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    elif isinstance(df, pl.DataFrame):
        return df.clone()
    elif isinstance(df, np.ndarray):
        return np.copy(df)
