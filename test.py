from pyfew.features.data_utils import load_data
from pyfew.features.core import extract_features
import pandas as pd

sample_rate = 50
window_length = 30
window_overlap = 15
data, my_times = load_data('/Users/hangy/Dphil/code/data.csv', window_length, sample_rate=sample_rate, window_overlap=window_overlap)

feats = [extract_features(epoch, sample_rate=sample_rate) for epoch in data]
feats = pd.DataFrame(feats)
print(feats.head())
