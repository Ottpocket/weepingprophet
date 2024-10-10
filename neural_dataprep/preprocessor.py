from typing import List, Dict, Any
from collections import defaultdict
import pickle
import os

class BasePreprocessor:
    """ 
    Preprocesses certain columns of a dataframe by 
        1) creating a FeatureTransformer and 
        2) running an instance of FeatureTransformer on each column 
        
    USAGE
    -----------------------------
    #pandas or polars both work here
    df = pd.DataFrame(
        {
            'a':[1,np.nan,3],
            'b':[11,np.nan,33],
            'c':['one',np.nan,'two']
        }
    )
    num_preprocessor = BasePreprocessor(
        cols_to_transform = ['a','b'],
        name_of_transformer_class = StandardFeatureTransformer
    )
    df_num = num_preprocessor.fit_transform(df)
    
    cat_preprocessor = BasePreprocessor(
        cols_to_transform = ['c'],
        name_of_transformer_class = CategoricalFeatureTransformer
    )
    df_cat = cat_preprocessor.fit_transform(df)
    """

    def __init__(self, cols_to_transform: List[str], name_of_transformer_class:BaseFeatureTransformer):
        self.transformer_dict = {col: name_of_transformer_class.factory_method() for col in cols_to_transform}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'BasePreprocessor':
        """Trains the preprocessors."""
        for col in self.transformer_dict.keys():
            self.transformer_dict[col].fit(df[col])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uses the preprocessor to transform data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        
        results = copy_dataframe(df[list(self.transformer_dict.keys())])
        #polars data
        if type(df) ==pl.dataframe.frame.DataFrame:
            
            return results.with_columns(
                [
                    transformer.transform(results[col]).alias(col)
                    for col, transformer in self.transformer_dict.items()
                ]
            )

        else:
            for col, transformer in self.transformer_dict.items():
                print(col, transformer)
                results[col] = transformer.transform(results[col])

            return results
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the preprocessor and transforms the data."""
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> None:
        """
        Saves the preprocessor object to a file.
        
        Args:
            filepath (str): The path where the preprocessor will be saved.
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BasePreprocessor':
        """
        Loads a preprocessor object from a file.
        
        Args:
            filepath (str): The path from which to load the preprocessor.
        
        Returns:
            BasePreprocessor: The loaded preprocessor object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
        
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor
