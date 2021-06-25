# Pyfew
A simple and light-weight Feature Extractor for Wearable accelerometer data. 
It is designed to quickly extract a rich set of well-selected spatial and frequency domain features
for tri-axial accelerometer. The extracted features can be easily incorporated in machine learning models. 
We also include several models for different detection tasks to illustrate how the extracted features can be used.

* Test icon (TODO)
* PyPI upload (TODO)

Github actions:
flake8 style check

# Dependencies

* Catch22
* Numpy
* Scipy

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

#### Detection