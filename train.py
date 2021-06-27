#!/usr/bin/env python3
from libs import *
import pandas as pd
import numpy as np


df = pd.DataFrame()
try:
    df = pd.read_csv("./data/data.csv")
except OSError:
    snackbar("Error: can\'t open data file (data/data.csv)", 'error')
except (ValueError, IndexError):
    snackbar("Error: not valid data file (data/data.csv)", 'error')
except Exception as err:
    snackbar("Unknown error", 'error')

if (df.shape[1] != 2):
    snackbar("There are two features missing from your dataset", 'error')

x, y = np.array(df.km), np.array(df.price)

model = SlyLinearRegression(0, 0)
model.fit(x, y)

df_model = pd.DataFrame({'b': [model.b], 'k': [model.k]})

df_model.to_csv('./model.csv', index=False)
