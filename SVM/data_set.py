from stock_data import get_period_date
import jqdatasdk as jq
import numpy as np
import pandas as pd
import pickle
from data_read_csv import to_csv

jq.auth('18813173861', 'wuyifan1994')
peroid = 'M'
start_date = '2010-01-01'
end_date = '2018-01-01'
industry_old_code = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', '801140',
                     '801150', \
                     '801160', '801170', '801180', '801200', '801210', '801230']
industry_new_code = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', '801140',
                     '801150', \
                     '801160', '801170', '801180', '801200', '801210', '801230', '801710', '801720', '801730', '801740',
                     '801750', \
                     '801760', '801770', '801780', '801790', '801880', '801890']

dateList = get_period_date(peroid, start_date, end_date)


def data_set(dateList, factor_data,n,train_set,train = True):
    if train:
        b,s = (0,12*n)
    else:
        b,s = (12*n,-1)
    for date in dateList[b:s]:
        train_df = factor_data[date]
        stock_list = list(train_df.index)
        data_close = jq.get_price(stock_list, date, dateList[dateList.index(date) + 1], '1d', 'close')['close']
        train_df['pchg'] = data_close.iloc[-1] / data_close.iloc[0] - 1
        train_df = train_df.dropna()
        train_df.sort_values(by='pchg', ascending=False, inplace=True)
        train_df_len = int(len(train_df['pchg']))
        train_df = train_df.iloc[:int(train_df_len / 10 * 3), ].append(train_df.iloc[int(train_df_len / 10 * 7):, :])
        mean = np.mean(train_df['pchg'])
        train_df['label'] = list(train_df['pchg'].apply(lambda x: 1 if x > mean else 0))
        if train:
            if train_set.empty:
                train_set = train_df
            else:
                train_set = train_set.append(train_df)
        else:
            train_set[date] = train_df
    return train_set
with open("./factor_solve_data.pkl", 'rb') as f:
    # pkl_file_read = read_file("factor_solve_data.pkl")
    #     factor_data = pickle.loads(StringIO(pkl_file_read))
    factor_data = pickle.load(f, encoding='iso-8859-1')
print(factor_data.keys())
train_set = pd.DataFrame()
train_set = data_set(dateList,factor_data,4,train_set,True)
train_set.to_csv("train_set.csv")
# test_set = dict()
# test_set = data_set(dateList,factor_data,4,test_set,False)
#
# save = pickle.dumps(test_set)
# with open("test_set.pkl",'wb+') as f:
#     f.write(save)



