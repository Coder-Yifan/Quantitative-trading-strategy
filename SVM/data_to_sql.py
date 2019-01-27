import pickle
import pymysql
from sqlalchemy import create_engine
from _mysql_connector import date_to_mysql
with open("./factor_solve_data.pkl", 'rb') as f:
    # pkl_file_read = read_file("factor_solve_data.pkl")
    #     factor_data = pickle.loads(StringIO(pkl_file_read))
    factor_data = pickle.load(f, encoding='iso-8859-1')
print(list(factor_data.keys()))
dates = list(factor_data.keys())
conn = create_engine('mysql+pymysql://root:123456@localhost:3306/stock_data?charset=utf8')
for date in dates:
    #print(factor_data[date])

    data = factor_data[date]
    data.reset_index(drop = False,inplace =True)
    data.rename(columns = {'index':'stock'},inplace = True)
    data.to_sql('factor_data', con=conn, if_exists='append', index=True)




