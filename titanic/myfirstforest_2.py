import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_fn = 'train.csv'
test_fn = 'test.csv'

train_data = pd.read_csv(train_fn)
test_data = pd.read_csv(test_fn)