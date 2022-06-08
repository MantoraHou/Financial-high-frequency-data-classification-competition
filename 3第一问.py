import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
## 数据导入 
print('Load data...')
df_train = pd.read_csv('./data/Y_train_run.csv') 
df_test = pd.read_csv('./data/Y_test_run.csv')

y_train = df_train.iloc[:, 7]	
y_test = df_test.iloc[:, 7]		

x_train = df_train.iloc[:, 8:]
x_test = df_test.iloc[:, 8:]


sc = StandardScaler()	
x_train = sc.fit_transform(x_train)	
x_test = sc.transform(x_test)

## 模型参数构建和训练
d_train = lgb.Dataset(x_train, label=y_train)	
params = {}	
# params['learning_rate'] = 0.05	
# params['boosting_type'] = 'gbdt'	
# params['objective'] = 'binary'	
# params['metric'] = 'binary_logloss'	
# params['sub_feature'] = 0.5	
# params['num_leaves'] = 95
# params['min_data'] = 91	
# params['max_depth'] = 3
# # params['min_data_in_leaf'] = 300
# params['L1 regularization'] = 0.01
params['objective'] = 'binary'	  # 	要用的算法
params['metric'] = 'binary_logloss'	 # mae: mean absolute error ， mse: mean squared error ， binary_logloss: loss for binary classification ， multi_logloss: loss for multi classification


# 第一步 学习率和迭代次数
params['learning_rate'] = 0.05	
params['boosting_type'] = 'gbdt'
params['metric'] = 'auc'

params['bagging_fraction']= 0.7 # 数据采样
params['feature_fraction']= 0.8  #例如 为0.8时，意味着在每次迭代中随机选择80％的参数来建树	boosting 为 random forest 时用
params['bagging_freq']= 5 #


# 第二步：确定max_depth和num_leaves

params['num_leaves'] = 95  #取值应 <= 2 ^（max_depth）， 超过此值会导致过拟合
params['max_depth'] = 3

#第三步：确定min_data_in_leaf和max_bin in
params['min_data_in_leaf'] = 91  #叶子可能具有的最小记录数	默认20，过拟合时用
params['max_bin'] = 85	

#第四步：确定feature_fraction、bagging_fraction、bagging_freq

#第五步：确定lambda_l1和lambda_l2
params['L2 regularization'] = 0.9
params['L1 regularization'] = 0.6

# 第六步：确定 min_split_gain 
params['min_split_gain'] = 1.0#描述分裂的最小 gain	控制树的有用的分裂

#######################训练集的预测偏差################
print('='*30,'训练集的预测偏差','='*30)
clf = lgb.train(params, d_train, 100)
y_pred=clf.predict(x_train)	

y_mean = np.mean(y_pred) # 均值
y_std = np.std(y_pred,ddof=1)  # 标准差


# 对数据进行归一化处理
for i in range(0,145333):	
    if (y_pred[i]-y_mean)/y_std>=0.2:       # 将预测数据转为0，1变量
       y_pred[i]=1	
    else:
       y_pred[i]=0

print('均方根误差为:', mean_squared_error(y_train, y_pred) ** 0.5) # 计算真实值和预测值之间的均方根误差



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
    if (y_pred[i]-y_mean)/y_std>=0.5:       	
       y_pred[i]=1	
    else:
       y_pred[i]=0


print('均方根误差是：', mean_squared_error(y_test, y_pred) ** 0.5) # 计算真实值和预测值之间的均方根误差


# 混淆矩阵

cm = confusion_matrix(y_test, y_pred)	
print(cm)

# 准确率计算	

accuracy=accuracy_score(y_pred,y_test)
print(accuracy)


#  准确率，召回率，f1-score

print(classification_report(y_test, y_pred, target_names=None))
