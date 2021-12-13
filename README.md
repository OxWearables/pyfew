[![Actions Status](https://github.com/activityMonitoring/pyfew/workflows/Pylint/badge.svg)](https://github.com/activityMonitoring/pyfew/actions)
[![Actions Status](https://github.com/activityMonitoring/pyfew/workflows/build/badge.svg)](https://github.com/activityMonitoring/pyfew/actions)

# Pyfew
A simple and light-weight Feature Extractor for Wearable accelerometer data. 
It is designed to quickly extract a rich set of well-selected spatial and frequency domain features
for tri-axial accelerometer. The extracted features can be easily incorporated in machine learning models. 
We also include several models for different detection tasks to illustrate how the extracted features can be used.

### Dependencies

* Catch22
* Numpy
* Scipy
* Yaml

### Install from pip
```bash
pip install pyfew
```

### Installation for Development
```bash
git clone git@github.com:activityMonitoring/pyfew.git
cd pywear
pip install .
```

### Examples 

#### Feature extraction
The csv data format should be like the following. 
```bash
                     time         x         y         z     T
0 2014-05-07 13:29:51.000  0.351895 -0.444269  0.922479  20.0
1 2014-05-07 13:29:51.020  0.287764 -0.402277  0.813471  20.0
2 2014-05-07 13:29:51.040  0.330980 -0.313290  0.862041  20.0
3 2014-05-07 13:29:51.060  0.428043 -0.325817  0.866287  20.0
4 2014-05-07 13:29:51.080  0.374533 -0.374217  0.859578  20.0
```

The table is a sample output of [actipy](https://github.com/activityMonitoring/actipy). Actipy allows one to parse any
compressed wearable data such as Gt3X and AX3 into a usable format. If you want to directly extract the features, 
data is a npy file of shape: `N x m x 3`. N is the number of samples. `M = epoch_len (sec) x sample_rate`, then the generated
output will be of shape `N x num_feats`.

```python
from pyfew.features.data_utils import load_data
from pyfew.features.core import extract_features

sample_rate = 50
window_length = 30
window_overlap = 15
data, my_times = load_data('/mini_data.csv',
                           window_length, 
                           sample_rate=sample_rate, 
                           window_overlap=window_overlap)
feats = extract_features(data, feature_set='full', sample_rate=sample_rate)
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
feats = extract_features(data, custom_features=custom_features, feature_set='full', sample_rate=sample_rate)
```

#### Feature set
You can also specify the set of features that you wanna use by specifying the `feature_set` argument. At the moment we 
support `minimal`, `default` and `full`. More sets will be included in the future.


