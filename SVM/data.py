import datetime
import pandas as pd
import numpy as np
from data_read_csv import to_csv
import pickle



from stock_data import get_period_date
from data_set import data_set
#使用pickle模块将数据对象保存到文件
with open("./factor_solve_data.pkl", 'rb') as f:
    # pkl_file_read = read_file("factor_solve_data.pkl")
    #     factor_data = pickle.loads(StringIO(pkl_file_read))
    factor_data = pickle.load(f, encoding='iso-8859-1')
print(factor_data.keys())



#
#print(factor_data['2009-12-31'].head())
peroid='M'
start_date='2010-01-01'
end_date='2018-01-01'
industry_old_code=['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150',\
                    '801160','801170','801180','801200','801210','801230']
industry_new_code=['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150',\
                    '801160','801170','801180','801200','801210','801230','801710','801720','801730','801740','801750',\
                   '801760','801770','801780','801790','801880','801890']

dateList=list(factor_data.keys())
train_set = pd.DataFrame()
test_set = dict()
train_set = data_set(dateList,factor_data,4,train_set,True)
test_set = data_set(dateList,factor_data,4,test_set,False)
to_csv("train_set.csv",train_set)
save = pickle.dumps(test_set)
with open("test_set.pkl",'wb+') as f:
    f.write(save)


