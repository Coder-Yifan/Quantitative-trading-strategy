import datetime
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels import regression
import pickle as pickle
from six import StringIO
# 导入pca
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sns

import jqdatasdk as jq

jq.auth('18813173861', 'wuyifan1994')
start_date = "2010-01-01"
end_date = "2010-02-01"


def get_period_date(period, start_date, end_date):
    """该函数为获取输入日期与输出日期之间的时间列表,
    设定转换周期period_type  转换为周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'"""
    stock_data = jq.get_price('000001.XSHE', start_date, end_date, 'daily', fields=['close'])
    stock_data['date'] = stock_data.index
    period_data = stock_data.resample(period,how = 'last')
    date = period_data.index
    pydate_array = date.to_pydatetime()
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)
    date_only_series = pd.Series(date_only_array)
    start_date = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    start_date = start_date + datetime.timedelta(days=-1)
    start_date = start_date.strftime("%Y-%m-%d")
    date_list = date_only_series.values.tolist()
    date_list.insert(0,start_date)
    return date_list

#get_period_date("W",start_date,end_date)


def delect_stop(stocks,begin_date,n=30*3):
    stock_list = list()
    begin_date = datetime.datetime.strptime(begin_date,"%Y-%m-%d")
    for stock in stocks:
        start_date = jq.get_security_info(stock).start_date
        if start_date < (begin_date - datetime.timedelta(days=n)).date():
            stock_list.append(stock)

    return stock_list

#获取股票池
def get_stock(stockPool,begin_date):
    if stockPool=='HS300':
        stockList=jq.get_index_stocks('000300.XSHG',begin_date)
    elif stockPool=='ZZ500':
        stockList=jq.get_index_stocks('399905.XSHE',begin_date)
    elif stockPool=='ZZ800':
        stockList=jq.get_index_stocks('399906.XSHE',begin_date)   
    elif stockPool=='CYBZ':
        stockList=jq.get_index_stocks('399006.XSHE',begin_date)
    elif stockPool=='ZXBZ':
        stockList=jq.get_index_stocks('399005.XSHE',begin_date)
    elif stockPool=='A':
        stockList=jq.get_index_stocks('000002.XSHG',begin_date)+jq.get_index_stocks('399107.XSHE',begin_date)
    #剔除ST股
    st_data=jq.get_extras('is_st',stockList, count = 1,end_date=begin_date)
    stockList = [stock for stock in stockList if not st_data[stock][0]]
    #剔除停牌、新股及退市股票
    stockList=delect_stop(stockList,begin_date)
    #print(stockList)
    return stockList
#get_stock('HS300','2017-06-01')

def winsorize_med(factor_data):
    """去极值函数"""
    mean = np.mean(factor_data, axis=0)
    std = np.std(factor_data, axis=0)
    # print(mean['open'])
    for column in factor_data.columns:
        discrate1 = mean[column] + 3*std[column]
        discrate2 = mean[column] - 3*std[column]
        for factor in range(0, len(factor_data[column])):
            if factor_data[column][factor] > discrate1:
                factor_data[column][factor] = discrate1
            elif factor_data[column][factor] < discrate2:
                factor_data[column][factor] = discrate2
    return factor_data
def data_preprocessing(factor_data,stock_list,industry_code,date):
    #去极值
    factor_data = winsorize_med(factor_data)
    #缺失值处理
    pass

def get_factor_data(stock_list,date):
    df = jq.query()





