#导入pca
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
train_data = pd.read_csv('train_set.csv')
target = train_data['label']
train = train_data.copy()
del train['pchg']
del train['label']

regressor = svm.SVC()
parameters = {'kernel':['rbf'],'C':[0.01,0.03,0.1,0.3,1,3,10],\
              'gamma':[1e-4,3e-4,1e-3,3e-3,0.01,0.03,0.1,0.3,1]}
clf = GridSearchCV(regressor,parameters,scoring='roc_auc',cv = 10)
train = np.array(train)
target = np.array(target)
clf.fit(train,target)
# 输出交叉验证的结果统计列表
#print (clf.best_score_)
# 输出每个模型的结果
print (clf.grid_scores_)
# 输出最佳模型结果
print (clf.best_score_)
# 输出最佳模型参数
print (clf.best_params_)
