import pickle
import sys
import pandas as pd
# Monkey patch for older pandas pickles
import pandas.core.indexes.base
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pandas.Index
import easydict
import bidict

with open("../PandemicLLM/data/processed_v5_4.pkl", 'rb') as f:
    data = pickle.load(f)

df = data.sta_dy_aug_data
print("Columns in df:")
print(df.columns.tolist())
print("\nFirst row sample:")
print(df.iloc[0])
