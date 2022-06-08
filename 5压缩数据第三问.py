import datetime
import pandas as pd
import numpy as np

import datetime

## 读取数据
df = pd.read_csv('./data/Y_train_run.csv')
## 将字符串设置时间戳
df['time']=pd.to_datetime(df['time'])
df['time']=pd.to_datetime(df['time'],format='%d-%m-%Y %H:%M')
df.index=df['time']



# #按1天聚合,取平均值
df=df.resample('B').mean()#  B 工作日
print(type(df))
df=df.dropna()  #删除有空值的行
df.to_csv('./data/Y_train_run_3.csv')





