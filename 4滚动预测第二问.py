
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print('数据载入')
df_train = pd.read_csv('./data/Y_train_run.csv')
df_test = pd.read_csv('./data/Y_test_run.csv')

df_train = pd.concat([df_train,df_test]) 


y_train = df_train.iloc[:,7]	
y_test = df_test.iloc[:, 7]	

x_train = df_train.iloc[:, 8:]
x_test = df_test.iloc[:, 8:]


sc = StandardScaler()	
x_train = sc.fit_transform(x_train)	
x_test = sc.transform(x_test)

## 模型参数构建和训练
d_train = lgb.Dataset(x_train, label=y_train)	
params = {}	
params['learning_rate'] = 0.05	
params['boosting_type'] = 'gbdt'	
params['objective'] = 'binary'	
params['metric'] = 'binary_logloss'	
	
params['num_leaves'] = 40
# params['min_data'] = 40	
params['max_depth'] = 8
params['min_data_in_leaf'] = 100
params['L1 regularization'] = 0.005
params['L1 regularization'] = 0.003



#######################训练集的预测偏差################
print('='*30,'训练集的预测偏差','='*30)
clf = lgb.train(params, d_train, 100)
y_pred=clf.predict(x_train)	

y_mean = np.mean(y_pred) # 均值
y_std = np.std(y_pred,ddof=1)  # 标准差


# 对数据进行归一化处理
for i in range(0,148184):	
    if (y_pred[i]-y_mean)/y_std>=0:       
       y_pred[i]=1	
    else:
       y_pred[i]=0

print('均方根误差是:', mean_squared_error(y_train, y_pred) ** 0.5) # 计算真实值和预测值之间的均方根误差


#混淆矩阵

cm = confusion_matrix(y_train, y_pred)	
print(cm)

#准确率计算
accuracy=accuracy_score(y_pred,y_train)
print(accuracy)


#  准确率，召回率，f1-score

print(classification_report(y_train, y_pred, target_names=None))

#######################测试集的预测偏差################
print('='*30,'测试集的预测偏差','='*30)
clf = lgb.train(params, d_train, 100)
# 模型预测

y_pred=clf.predict(x_test)	

y_mean = np.mean(y_pred) # 均值
y_std = np.std(y_pred,ddof=1)  # 标准差


# 对数据进行归一化处理
for i in range(0,2851):	  # 滚动预测中去掉了最后一个缺失值 
    if (y_pred[i]-y_mean)/y_std>=0:       
       y_pred[i]=1	
    else:
       y_pred[i]=0

print('均方根误差是：', mean_squared_error(y_test, y_pred) ** 0.5) # 计算真实值和预测值之间的均方根误差


#混淆矩阵

cm = confusion_matrix(y_test, y_pred)	
print(cm)

#准确率计算
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)



print(classification_report(y_test, y_pred, target_names=None))








