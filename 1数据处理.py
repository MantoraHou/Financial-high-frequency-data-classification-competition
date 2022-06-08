import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime







excel_folder='./data/B题数据集'

# 判断数据文件选取因变量
excel_files = os.listdir(excel_folder) # 导出excel文件名
len_finnal = 10000000
for excel_file in excel_files:
    ## 读取文件
    excel_path = os.path.join(excel_folder, excel_file) # 确定excel的地址
    df = pd.read_csv(excel_path)
    excel_name = excel_file.split(".")[0]

    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S',errors='ignore')  # 将字符串变为时间日期

    ## 划分训练集
    start = datetime.datetime(2018, 1, 1, 0, 0, 0)
    end = datetime.datetime(2021,9,1,0,0,0)
    subset = df[df['time']>start]
    data1 = subset[subset['time']<end]  


    ## 选取数据集长度最短的股票的收盘价为因变量
    len_new = len(data1.iloc[:,0]) # 获得新文件data1的因变量长度

    if len_new<len_finnal:  #将长度作比较
        len_finnal=len_new
        choesenfilepath=excel_path  # 江中药业为最终因变量选择
    else:
        continue
    



print(choesenfilepath)
###  因变量、自变量的选取       
## 因变量
df = pd.read_csv(choesenfilepath)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S',errors='ignore')  # 将字符串变为时间日期
## 划分训练集
start = datetime.datetime(2018, 1, 1, 0, 0, 0)
end = datetime.datetime(2021,9,1,0,0,0)
subset = df[df['time']>start]
Y_train = subset[subset['time']<end]  

## 划分测试集
start = datetime.datetime(2021, 9, 1, 0, 0, 0)
end = datetime.datetime(2021,9,19,0,0,0)
subset = df[df['time']>start]
Y_test = subset[subset['time']<end]
    
excel_files = os.listdir(excel_folder) # 导出excel文件名
for excel_file in excel_files:
    ## 读取文件
    excel_path = os.path.join(excel_folder, excel_file) # 确定excel的地址
  
    # 自变量
    if excel_path == choesenfilepath:
       continue
    else:
        df = pd.read_csv(excel_path)
        excel_name = excel_file.split(".")[0]

        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S',errors='ignore')  # 将字符串变为时间日期
        ## 划分训练集，自变量添加入训练集
        start = datetime.datetime(2018, 1, 1, 0, 0, 0)
        end = datetime.datetime(2021,9,1,0,0,0)
        subset = df[df['time']>start]
        data1 = subset[subset['time']<end]  

        Y_train=pd.merge(Y_train, data1, on='time')  # 以时间为基准，将自变量添入训练集
        ## 划分测试集,自变量添加入测试集
        start = datetime.datetime(2021, 9, 1, 0, 0, 0)
        end = datetime.datetime(2021,9,19,0,0,0)
        subset = df[df['time']>start]
        data2 = subset[subset['time']<end]

        Y_test=pd.merge(Y_test, data2, on='time')  # 以时间为基准，将自变量添入测试集



Y_train.to_csv('./data/Y_train.csv')
Y_test.to_csv('./data/Y_test.csv')

print(len(Y_train))  # 145334
print(len(Y_test))   # 2852
