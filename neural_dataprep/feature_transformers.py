"""

"""
from abc import ABC, abstractmethod
from series_interface import SeriesFactory
class BaseFeatureTransformer(ABC):
    """ 
    Base class for feature transformers that operate on a single column of a dataframe 
    
    USAGE
    --------------------------------------------
    #Works with pandas, polars, or numpy
    df = pl.DataFrame({'a': ['b' if (i%2) else None for i in range(11)]})
    cft = CategoricalFeatureTransformer()
    cft.fit_transform(df['a']) 
    """

    def __init__(self):
        self.is_fitted = False
        self.series_interface = None

    def fit(self, series) -> 'BaseFeatureTransformer':
        self.series_interface = SeriesFactory.get_interface(series)
        self._fit_hook(series)
        self.is_fitted = True
        return self
    
    def fit_transform(self,series):
        return self.fit(series).transform(series)

    def transform(self, series):
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform.")
        series = self._transform_hook(series)
        return series

    @abstractmethod
    def _fit_hook(self, series):
        pass

    def _transform_hook(self, series):
        return series

    @classmethod
    def factory_method(cls):
        """ Creates a new instance of the transformer """
        return cls()

class NormalizingFeatureTransformer(BaseFeatureTransformer):
    """ 
    Normalizes a series using (col - mean) / std 
    
    USAGE
    -----------------------------
    #works for polars, pandas, or numpy
    df = pl.DataFrame({'a': [i if (i%2) else np.nan for i in range(11)]})
    sft = NormalizingFeatureTransformer()
    sft.fit_transform(df['a'])
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def _fit_hook(self, series):
        self.mean = self.series_interface.mean(series)
        self.std = self.series_interface.std(series)

    def _transform_hook(self, series):
        normalized_series = (series - self.mean) / self.std
        return self.series_interface.fillna(normalized_series, value=0.)
    
    
class StandardFeatureTransformer(BaseFeatureTransformer):
    """ 
    Standardizes a series using (max - value) / (max - min) 
    
    USAGE
    -----------------------------
    #works for polars, pandas, or numpy
    df = pl.DataFrame({'a': [i if (i%2) else np.nan for i in range(11)]})
    sft = StandardFeatureTransformer()
    sft.fit_transform(df['a'])
    """

    def __init__(self):
        super().__init__()
        self.max_value = None
        self.min_value = None

    def _fit_hook(self, series):
        self.max_value = self.series_interface.max(series)
        self.min_value = self.series_interface.min(series)

    def _transform_hook(self, series):
        if self.max_value == self.min_value:
            # Handle the case where all values are the same
            return self.series_interface.fillna(series, value=0.)
        
        standardized_series = (series - self.min_value) / (self.max_value - self.min_value)
        return self.series_interface.fillna(standardized_series, value=0.)


##################################################################
#Interface Helper functions
##################################################################
def copy_dataframe(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    elif isinstance(df, pl.DataFrame):
        return df.clone()
    elif isinstance(df, np.ndarray):
        return np.copy(df)
    
class CategoricalFeatureTransformer(BaseFeatureTransformer):
    """ 
    Changes a categorical columns to integers 0,1,2,... 
    
    
    USAGE
    --------------------------------------------
    #Works with pandas, polars, or numpy
    df = pl.DataFrame({'a': ['b' if (i%2) else None for i in range(11)]})
    cft = CategoricalFeatureTransformer()
    cft.fit_transform(df['a']) 
    """

    def __init__(self, threshhold=2):
        super().__init__()
        self.threshhold = threshhold
        self.transform_dict = None

    def _fit_hook(self, series):
        vc = self.series_interface.value_counts(series) #returns a pandas dataframe
        col_name = [col for col in vc.columns if col != 'count'][0]
        transform_dict = vc.loc[vc['count'] > self.threshhold, col_name].reset_index(drop=True).to_dict()
        self.transform_dict = {value:key for key,value in transform_dict.items()}
    def _transform_hook(self, series):
        return self.series_interface.map_dict(series, self.transform_dict)
