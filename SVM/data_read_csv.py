import pandas as pd

def read_csv(path):
    return pd.read_csv('train_set.csv',index_col=0)
#data = pd.read_csv('train_set.csv',index_col=0)
def to_csv(path,data):

    data.reset_index(drop=False,inplace=True)
    data.rename(columns={'index':'stock'},inplace=True)
    print(data.head())
    data.to_csv("train_set.csv")
