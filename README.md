# weepingprophet

Stuff to help my neural networks.

#### To Use in Kaggle
```
#Clone this repo locally to kaggle notebook
!git clone https://github.com/Ottpocket/weepingprophet.git
```
```
#attach to system path then import
import sys
sys.path.append('weepingprophet')
from neural_dataprep.preprocessor import BasePreprocessor
from neural_dataprep.feature_transformers import StandardFeatureTransformer, CategoricalFeatureTransformer
```
```
#Usage
df = pd.DataFrame(
    {
        'a':[1,np.nan,3],
        'b':[11,np.nan,33],
        'c':['one',np.nan,'two']
    }
)

#process your numerical data via standardizing
num_preprocessor = BasePreprocessor(
    cols_to_transform = ['a','b'],
    name_of_transformer_class = StandardFeatureTransformer
)
df_num = num_preprocessor.fit_transform(df)

#process your categorical data via map strings to 0,1,2,...
cat_preprocessor = BasePreprocessor(
    cols_to_transform = ['c'],
    name_of_transformer_class = CategoricalFeatureTransformer
)
df_cat = cat_preprocessor.fit_transform(df)
```
