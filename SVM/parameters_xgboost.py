#导入pca
from sklearn.decomposition import PCA
#from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
train_data = pd.read_csv('train_set.csv',index_col=0)
train_y = train_data['label']
train_x = train_data.copy()
del train_x['pchg']
del train_x['label']


def modelMetrics(clf, train_x, train_y, isCv=True, cv_folds=5, early_stopping_rounds=50):
    #print("OK! start")
    if isCv:
        xgb_param = clf.get_xgb_params()

        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)  # 是否显示目前几颗树额
        clf.set_params(n_estimators=cvresult.shape[0])

    clf.fit(train_x, train_y, eval_metric='auc')

    # 预测
    train_predictions = clf.predict(train_x)
    train_predprob = clf.predict_proba(train_x)[:, 1]  # 1的概率

    # 打印
    print("n_estimators:",cvresult.shape[0])
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))
    feat_imp = pd.Series(clf.get_booster().get_fscore())
    feat_imp = feat_imp.sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature importance')
    plt.ylabel('Feature Importance Score')
    plt.show()


def tun_parameters(train_x,train_y):
    xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
                         colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
    modelMetrics(xgb1,train_x,train_y)
tun_parameters(train_x,train_y)

