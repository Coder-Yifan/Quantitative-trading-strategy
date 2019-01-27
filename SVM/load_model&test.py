import pickle
from sklearn import svm,metrics
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
#load_model
def model_load(path):
    with open(path,"rb")as f:
        content = f.read()
    return pickle.loads(content)
with open("./test_set.pkl", 'rb') as f:
    test_set = pickle.load(f, encoding='iso-8859-1')
#print(test_data['2014-05-31'].head())
print(test_set.keys())
dateList = list(test_set.keys())
path = "svm.model"
model = model_load(path)
#test_data = pd.read_csv("test_set.csv",index_col=0)
test_predict = dict()
test_score = list()
test_auc = list()
test_date = list()
for date in dateList:
    test_date.append(date)
    test_data = test_set[date]
    target = test_data['label']
    test = test_data.copy()
    del test['label']
    del test['pchg']
    test = np.array(test)
    target = np.array(target)
    predict = model.predict(test)
    test_predict[date]=predict
    test_score.append(model.score(test,target))
    #test_auc.append(metrics.roc_auc_score(test,predict))
print("测试集正确率：",np.mean(test_score))
#print("测试机AUC:",np.mean(test_auc))
# xs_date = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in test_date]
# ys_score = test_score
# # 配置横坐标
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#
# plt.plot(xs_date, ys_score,'r')
# # 自动旋转日期标记
# plt.gcf().autofmt_xdate()
# # 横坐标标记
# plt.xlabel('date')
# # 纵坐标标记
# plt.ylabel("test accuracy")
# plt.show()
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
factor_predict_corr = pd.DataFrame()
for date in dateList:
    test_feature = test_set[date].copy()
    del test_feature['pchg']
    del test_feature['label']
    test_feature['predict'] = list(test_predict[date])
    # print(test_feature.corr().head())
    factor_predict_corr[date] = test_feature.corr()['predict']
    # print("++++++++++++++")
    # print(factor_predict_corr[date].head())

factor_predict_corr = factor_predict_corr.iloc[:-1]
# 高斯核 SVM 模型对于下期涨跌预测值与本期因子值之间相关系数示意图
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
sns.set()
ax = sns.heatmap(factor_predict_corr)
plt.show()

