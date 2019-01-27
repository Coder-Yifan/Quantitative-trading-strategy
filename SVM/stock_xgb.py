import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
#from data_set import data_set,dateList
import pickle as pl

from xgboost import XGBClassifier
#import xgboost as xgb

train_data = pd.read_csv('train_set.csv',index_col=0)
target = train_data['label']
train = train_data.copy()
del train['pchg']
del train['label']
print("数据准备完毕")
m = 3
model = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=7,min_child_weight=1,gamma=0.1,subsample=0.8,
                         colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
train_score = []
test_score = []
train_auc = []
test_auc = []

for i in range(m):
    print("第{}次迭代".format(i))
    X_train,X_test,y_train,y_test = train_test_split(np.array(train),np.array(target))
    model.fit(X_train,y_train)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    score_train = model.score(X_train,y_train)
    score_test = model.score(X_test,y_test)
    roc_auc_score_train = metrics.roc_auc_score(y_train,train_predict)
    roc_auc_score_test = metrics.roc_auc_score(y_test,test_predict)
    print('第{}次迭代,样本内训练集正确率：{}'.format(i,score_train))
    print('第{}次迭代,交叉验证集正确率：{}'.format(i,score_test) )
    print('第{}次迭代,样本内训练集AUC：{}'.format(i,roc_auc_score_train))
    print('第{}次迭代,交叉验证集AUC：{}'.format(i,roc_auc_score_test))
    train_score.append(score_train)
    test_score.append(score_test)
    train_auc.append(roc_auc_score_train)
    test_auc.append(roc_auc_score_test)
save = pl.dumps(model)
with open("xgboost.model",'wb+') as f:
    f.write(save)
print ('样本内训练集正确率：',np.mean(train_score))
print ('交叉验证集正确率：',np.mean(test_score))
print ('样本内训练集AUC：',np.mean(train_auc))
print ('交叉验证集AUC：',np.mean(test_auc))




