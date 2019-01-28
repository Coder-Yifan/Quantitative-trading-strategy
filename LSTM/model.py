from keras import Sequential
import pandas as pd

data = pd.read_csv("data_set.csv",index_col=0)
data = data.iloc[:,2:].values


