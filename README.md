# Pyfew
A simple and light-weight Feature Extractor for Wearable accelerometer data. 
It is designed to quickly extract a rich set of well-selected spatial and frequency domain features
for tri-axial accelerometer. The extracted features can be easily incorporated in machine learning models. 
We also include several models for different detection tasks to illustrate how the extracted features can be used.

* Test icon (TODO)
* PyPI upload (TODO)

Github actions:
flake8 style check

### Dependencies

* Catch22
* Numpy
* Scipy
* Yaml


### Installation in Python
```bash
git clone git@github.com:activityMonitoring/pyfew.git
cd pywear
pip install .
```


### Examples 

#### Feature extraction
```python
from pyfew.features.data_utils import load_data
from pyfew.features.core import extract_features
import pandas as pd

sample_rate = 50
window_length = 30
window_overlap = 15
data, my_times = load_data('mini_data.csv', window_length, sample_rate=sample_rate, window_overlap=window_overlap)

feats = [extract_features(epoch, sample_rate=sample_rate) for epoch in data]
feats = pd.DataFrame(feats)
print(feats.head())
```

#### Custom features
If you would like to add your custom features, you can define the feature extraction functions and add them to the
current set of features easily like this:
```python
def sample_featureI(xyz, feats, feats_name='cf1'):
    feats[feats_name] = np.max(xyz, axis=0)
    return feats


def sample_featureII(xyz, feats, feats_name='cf2'):
    feats[feats_name] = np.min(xyz, axis=0)
    return feats

custom_features = [sample_featureI, sample_featureII]
feats = [extract_features(epoch, sample_rate=sample_rate, 
                          custom_features=custom_features) for epoch in data]
```
#### Feature set
You can also specify the set of features that you wanna use by specifying the `feature_set` argument. At the moment we 
support `minimal`, `default` and `full`. More sets will be included in the future.


### Detection