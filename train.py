import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

# from keras.models import Sequential
# from keras.layers import LSTM,Dense,Dropout

import tensorflow as tf
from tensorflow.keras import Sequential,layers,utils,losses
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard

# import torch
# import torch.nn as nn

data = pd.read_csv('./广州.csv')
zq_data = pd.read_csv('./肇庆.csv')
sz_data = pd.read_csv('./深圳.csv')
dw_data = pd.read_csv('./东莞.csv')
hz_data = pd.read_csv('./惠州.csv')
zs_data = pd.read_csv('./中山.csv')
jm_data = pd.read_csv('./江门.csv')
zh_data = pd.read_csv('./珠海.csv')
fs_data = pd.read_csv('./佛山.csv')

start = 1826
end = 3287
wins = 30
test_size = 30
batch_size = 256
epochs = 2000

del data['rank'],sz_data['rank'],hz_data['rank'],dw_data['rank'],zq_data['rank'],zs_data['rank'],fs_data['rank'],zh_data['rank'],jm_data['rank']

def IAQI(name,data):
    if(name == 'so2'):
        return np.round((1.0)*(data-0.0) + 0,2)
    if(name == 'co'):
        return np.round((50.0/5)*(data-0.0) + 0,2)
    if(name == 'no2'):
        if(data <= 40):
            return np.round(((50.0-0.0)/(40.-0))*(data-0.0) + 0,2)
        elif(data <= 80):
            return np.round(((100.0-50.0)/(80.-40))*(data-40.) + 50,2)
        else:
            return np.round(((150.0-100.0)/(180.-80.))*(data-80.0) + 100,2)
    if(name == 'pm2.5'):
        if(data <= 35):
            return np.round(((50.0-0.0)/(35.-0))*(data-0.0) + 0,2)
        elif(data <= 75):
            return np.round(((100.0-50.0)/(75.-35))*(data-35.) + 50,2)
        elif(data <= 115):
            return np.round(((150.0-100.0)/(115.-75.))*(data-75.0) + 100.,2)
        elif(data <= 150):
            return np.round(((200.0-150.0)/(150.-115.))*(data-115.0) + 150.,2)
        else:
            return np.round(((300.0-200.0)/(250.-150.))*(data-115.0) + 200.,2)
    if(name == 'pm10'):
        if(data <= 50):
            return np.round(((50.0-50.0)/(50.-0))*(data-0.0) + 0.,2)
        elif(data <= 150):
            return np.round(((100.0-50.0)/(150.-50))*(data-50.) + 50.0,2)
        elif(data <= 250):
            return np.round(((150.0-100.0)/(250.-150.))*(data-150.0) + 100.,2)
    if(name == 'o3'):
        if(data <= 100):
            return np.round(((50.0-0.0)/(100.-0))*(data-0.0) + 0,2)
        elif(data <= 160):
            return np.round(((100.0-50.0)/(160.-100.))*(data-100.) + 50,2)
        elif(data <= 215):
            return np.round(((150.0-100.0)/(215.-160.))*(data-160.0) + 100.,2)
        elif(data <= 265):
            return np.round(((200.0-150.0)/(265.-215.))*(data-215.0) + 150.,2)
        else:
            return np.round(((300.0-200.0)/(800.-265.))*(data-265.0) + 200.,2)

data['pm2.5_iaqi'] = data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
data['pm10_iaqi'] = data['pm10'].apply(lambda x : IAQI('pm10',x)).values
data['co_iaqi'] = data['co'].apply(lambda x : IAQI('co',x)).values
data['so2_iaqi'] = data['so2'].apply(lambda x : IAQI('so2',x)).values
data['no2_iaqi'] = data['no2'].apply(lambda x : IAQI('no2',x)).values
data['o3_iaqi'] = data['o3'].apply(lambda x : IAQI('o3',x)).values

sz_data['pm2.5_iaqi'] = sz_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
sz_data['pm10_iaqi'] = sz_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
sz_data['co_iaqi'] = sz_data['co'].apply(lambda x : IAQI('co',x)).values
sz_data['so2_iaqi'] = sz_data['so2'].apply(lambda x : IAQI('so2',x)).values
sz_data['no2_iaqi'] = sz_data['no2'].apply(lambda x : IAQI('no2',x)).values
sz_data['o3_iaqi'] = sz_data['o3'].apply(lambda x : IAQI('o3',x)).values

zq_data['pm2.5_iaqi'] = zq_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
zq_data['pm10_iaqi'] = zq_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
zq_data['co_iaqi'] = zq_data['co'].apply(lambda x : IAQI('co',x)).values
zq_data['so2_iaqi'] = zq_data['so2'].apply(lambda x : IAQI('so2',x)).values
zq_data['no2_iaqi'] = zq_data['no2'].apply(lambda x : IAQI('no2',x)).values
zq_data['o3_iaqi'] = zq_data['o3'].apply(lambda x : IAQI('o3',x)).values

dw_data['pm2.5_iaqi'] = dw_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
dw_data['pm10_iaqi'] = dw_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
dw_data['co_iaqi'] = dw_data['co'].apply(lambda x : IAQI('co',x)).values
dw_data['so2_iaqi'] = dw_data['so2'].apply(lambda x : IAQI('so2',x)).values
dw_data['no2_iaqi'] = dw_data['no2'].apply(lambda x : IAQI('no2',x)).values
dw_data['o3_iaqi'] = dw_data['o3'].apply(lambda x : IAQI('o3',x)).values

zs_data['pm2.5_iaqi'] = zs_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
zs_data['pm10_iaqi'] = zs_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
zs_data['co_iaqi'] = zs_data['co'].apply(lambda x : IAQI('co',x)).values
zs_data['so2_iaqi'] = zs_data['so2'].apply(lambda x : IAQI('so2',x)).values
zs_data['no2_iaqi'] = zs_data['no2'].apply(lambda x : IAQI('no2',x)).values
zs_data['o3_iaqi'] = zs_data['o3'].apply(lambda x : IAQI('o3',x)).values

hz_data['pm2.5_iaqi'] = hz_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
hz_data['pm10_iaqi'] = hz_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
hz_data['co_iaqi'] = hz_data['co'].apply(lambda x : IAQI('co',x)).values
hz_data['so2_iaqi'] = hz_data['so2'].apply(lambda x : IAQI('so2',x)).values
hz_data['no2_iaqi'] = hz_data['no2'].apply(lambda x : IAQI('no2',x)).values
hz_data['o3_iaqi'] = hz_data['o3'].apply(lambda x : IAQI('o3',x)).values

fs_data['pm2.5_iaqi'] = fs_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
fs_data['pm10_iaqi'] = fs_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
fs_data['co_iaqi'] = fs_data['co'].apply(lambda x : IAQI('co',x)).values
fs_data['so2_iaqi'] = fs_data['so2'].apply(lambda x : IAQI('so2',x)).values
fs_data['no2_iaqi'] = fs_data['no2'].apply(lambda x : IAQI('no2',x)).values
fs_data['o3_iaqi'] = fs_data['o3'].apply(lambda x : IAQI('o3',x)).values

jm_data['pm2.5_iaqi'] = jm_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
jm_data['pm10_iaqi'] = jm_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
jm_data['co_iaqi'] = jm_data['co'].apply(lambda x : IAQI('co',x)).values
jm_data['so2_iaqi'] = jm_data['so2'].apply(lambda x : IAQI('so2',x)).values
jm_data['no2_iaqi'] = jm_data['no2'].apply(lambda x : IAQI('no2',x)).values
jm_data['o3_iaqi'] = jm_data['o3'].apply(lambda x : IAQI('o3',x)).values

zh_data['pm2.5_iaqi'] = zh_data['pm2_5'].apply(lambda x : IAQI('pm2.5',x)).values
zh_data['pm10_iaqi'] = zh_data['pm10'].apply(lambda x : IAQI('pm10',x)).values
zh_data['co_iaqi'] = zh_data['co'].apply(lambda x : IAQI('co',x)).values
zh_data['so2_iaqi'] = zh_data['so2'].apply(lambda x : IAQI('so2',x)).values
zh_data['no2_iaqi'] = zh_data['no2'].apply(lambda x : IAQI('no2',x)).values
zh_data['o3_iaqi'] = zh_data['o3'].apply(lambda x : IAQI('o3',x)).values

data['time_point'] = pd.to_datetime(data['time_point'])
sz_data['time_point'] = pd.to_datetime(sz_data['time_point'])
zq_data['time_point'] = pd.to_datetime(zq_data['time_point'])
dw_data['time_point'] = pd.to_datetime(dw_data['time_point'])
zs_data['time_point'] = pd.to_datetime(zs_data['time_point'])
hz_data['time_point'] = pd.to_datetime(hz_data['time_point'])
fs_data['time_point'] = pd.to_datetime(fs_data['time_point'])
jm_data['time_point'] = pd.to_datetime(jm_data['time_point'])
zh_data['time_point'] = pd.to_datetime(zh_data['time_point'])
# data['time_point'] = pd.to_datetime(data['time_point'])

# data['year'] = data['time_point'].dt.year
data['month'] = data['time_point'].dt.month
data['quarter'] = data['time_point'].dt.quarter
data['dayofweek'] = data['time_point'].dt.dayofweek
del data['time_point']

# 污染程度
data.loc[data['quality'] == '优','quality'] = 1
data.loc[data['quality'] == '良','quality'] = 2
data.loc[data['quality'] == '轻度污染','quality'] = 3
data.loc[data['quality'] == '中度污染','quality'] = 4
data.loc[data['quality'] == '重度污染','quality'] = 5

data = pd.get_dummies(data,columns=['quarter','month','dayofweek'])

sz_data['month'] = sz_data['time_point'].dt.month
sz_data['quarter'] = sz_data['time_point'].dt.quarter
sz_data['dayofweek'] = sz_data['time_point'].dt.dayofweek
del sz_data['time_point']

# 污染程度
sz_data.loc[sz_data['quality'] == '优','quality'] = 1
sz_data.loc[sz_data['quality'] == '良','quality'] = 2
sz_data.loc[sz_data['quality'] == '轻度污染','quality'] = 3
sz_data.loc[sz_data['quality'] == '中度污染','quality'] = 4
sz_data.loc[sz_data['quality'] == '重度污染','quality'] = 5

sz_data = pd.get_dummies(sz_data,columns=['quarter','month','dayofweek'])

dw_data['month'] = dw_data['time_point'].dt.month
dw_data['quarter'] = dw_data['time_point'].dt.quarter
dw_data['dayofweek'] = dw_data['time_point'].dt.dayofweek
del dw_data['time_point']

# 污染程度
dw_data.loc[dw_data['quality'] == '优','quality'] = 1
dw_data.loc[dw_data['quality'] == '良','quality'] = 2
dw_data.loc[dw_data['quality'] == '轻度污染','quality'] = 3
dw_data.loc[dw_data['quality'] == '中度污染','quality'] = 4
dw_data.loc[dw_data['quality'] == '重度污染','quality'] = 5
dw_data.loc[dw_data['quality'] == '严重污染','quality'] = 6

dw_data = pd.get_dummies(dw_data,columns=['quarter','month','dayofweek'])

zq_data['month'] = zq_data['time_point'].dt.month
zq_data['quarter'] = zq_data['time_point'].dt.quarter
zq_data['dayofweek'] = zq_data['time_point'].dt.dayofweek
del zq_data['time_point']

# 污染程度
zq_data.loc[zq_data['quality'] == '优','quality'] = 1
zq_data.loc[zq_data['quality'] == '良','quality'] = 2
zq_data.loc[zq_data['quality'] == '轻度污染','quality'] = 3
zq_data.loc[zq_data['quality'] == '中度污染','quality'] = 4
zq_data.loc[zq_data['quality'] == '重度污染','quality'] = 5
zq_data.loc[zq_data['quality'] == '严重污染','quality'] = 6

zq_data = pd.get_dummies(zq_data,columns=['quarter','month','dayofweek']) 


zs_data['month'] = zs_data['time_point'].dt.month
zs_data['quarter'] = zs_data['time_point'].dt.quarter
zs_data['dayofweek'] = zs_data['time_point'].dt.dayofweek
del zs_data['time_point']

# 污染程度
zs_data.loc[zs_data['quality'] == '优','quality'] = 1
zs_data.loc[zs_data['quality'] == '良','quality'] = 2
zs_data.loc[zs_data['quality'] == '轻度污染','quality'] = 3
zs_data.loc[zs_data['quality'] == '中度污染','quality'] = 4
zs_data.loc[zs_data['quality'] == '重度污染','quality'] = 5

zs_data = pd.get_dummies(zs_data,columns=['quarter','month','dayofweek']) 

hz_data['month'] = hz_data['time_point'].dt.month
hz_data['quarter'] = hz_data['time_point'].dt.quarter
hz_data['dayofweek'] = hz_data['time_point'].dt.dayofweek
del hz_data['time_point']

# 污染程度
hz_data.loc[hz_data['quality'] == '优','quality'] = 1
hz_data.loc[hz_data['quality'] == '良','quality'] = 2
hz_data.loc[hz_data['quality'] == '轻度污染','quality'] = 3
hz_data.loc[hz_data['quality'] == '中度污染','quality'] = 4
hz_data.loc[hz_data['quality'] == '重度污染','quality'] = 5

hz_data = pd.get_dummies(hz_data,columns=['quarter','month','dayofweek'])

fs_data['month'] = fs_data['time_point'].dt.month
fs_data['quarter'] = fs_data['time_point'].dt.quarter
fs_data['dayofweek'] = fs_data['time_point'].dt.dayofweek
del fs_data['time_point']

# 污染程度
fs_data.loc[fs_data['quality'] == '优','quality'] = 1
fs_data.loc[fs_data['quality'] == '良','quality'] = 2
fs_data.loc[fs_data['quality'] == '轻度污染','quality'] = 3
fs_data.loc[fs_data['quality'] == '中度污染','quality'] = 4
fs_data.loc[fs_data['quality'] == '重度污染','quality'] = 5

fs_data = pd.get_dummies(fs_data,columns=['quarter','month','dayofweek']) 

jm_data['month'] = jm_data['time_point'].dt.month
jm_data['quarter'] = jm_data['time_point'].dt.quarter
jm_data['dayofweek'] = jm_data['time_point'].dt.dayofweek
del jm_data['time_point']

# 污染程度
jm_data.loc[jm_data['quality'] == '优','quality'] = 1
jm_data.loc[jm_data['quality'] == '良','quality'] = 2
jm_data.loc[jm_data['quality'] == '轻度污染','quality'] = 3
jm_data.loc[jm_data['quality'] == '中度污染','quality'] = 4
jm_data.loc[jm_data['quality'] == '重度污染','quality'] = 5

jm_data = pd.get_dummies(jm_data,columns=['quarter','month','dayofweek']) 

zh_data['month'] = zh_data['time_point'].dt.month
zh_data['quarter'] = zh_data['time_point'].dt.quarter
zh_data['dayofweek'] = zh_data['time_point'].dt.dayofweek
del zh_data['time_point']

# 污染程度
zh_data.loc[zh_data['quality'] == '优','quality'] = 1
zh_data.loc[zh_data['quality'] == '良','quality'] = 2
zh_data.loc[zh_data['quality'] == '轻度污染','quality'] = 3
zh_data.loc[zh_data['quality'] == '中度污染','quality'] = 4
zh_data.loc[zh_data['quality'] == '重度污染','quality'] = 5

zh_data = pd.get_dummies(zh_data,columns=['quarter','month','dayofweek']) 

X = data.loc[start:end,data.columns != 'aqi']
y = data.loc[start:end,data.columns == 'aqi']

sz_X = sz_data.loc[start:end,sz_data.columns != 'aqi']
sz_y = sz_data.loc[start:end,sz_data.columns == 'aqi']

zq_X = zq_data.loc[start:end,zq_data.columns != 'aqi']
zq_y = zq_data.loc[start:end,zq_data.columns == 'aqi']

dw_X = dw_data.loc[start:end,dw_data.columns != 'aqi']
dw_y = dw_data.loc[start:end,dw_data.columns == 'aqi']

zs_X = dw_data.loc[start:end,zs_data.columns != 'aqi']
zs_y = dw_data.loc[start:end,zs_data.columns == 'aqi']

hz_X = dw_data.loc[start:end,hz_data.columns != 'aqi']
hz_y = dw_data.loc[start:end,hz_data.columns == 'aqi']

zh_X = zh_data.loc[start:end,zh_data.columns != 'aqi']
zh_y = zh_data.loc[start:end,zh_data.columns == 'aqi']

jm_X = jm_data.loc[start:end,jm_data.columns != 'aqi']
jm_y = jm_data.loc[start:end,jm_data.columns == 'aqi']

fs_X = fs_data.loc[start:end,fs_data.columns != 'aqi']
fs_y = fs_data.loc[start:end,fs_data.columns == 'aqi']

def creat_dataset(X,y,seq_len=10,test_size=31):
    
    # 训练集
    X_train = X.iloc[:-test_size,:]
    y_train = y.iloc[:-test_size,:]

    # 测试集
    X_test = X.iloc[-test_size-seq_len:,:]
    y_test = y.iloc[-test_size-seq_len:,:]
    
    X_train_list = []
    y_train_list = []
    
    X_test_list = []
    y_test_list = []
    
    for i in range(0,len(X_train)-seq_len,1):
        X_train_list.append(X_train.iloc[i:i+seq_len])
        y_train_list.append(y_train.iloc[i+seq_len])
    for i in range(0,len(X_test)-seq_len,1):
        X_test_list.append(X_test.iloc[i:i+seq_len])
        y_test_list.append(y_test.iloc[i+seq_len])
    return np.array(X_train_list,dtype=float),np.array(y_train_list,dtype=float),np.array(X_test_list,dtype=float),np.array(y_test_list,dtype=float)

print("正在划分训练集和测试集")
X_train1,y_train1,X_test1,y_test1 = creat_dataset(X,y,seq_len=wins,test_size=test_size)

X_train2,y_train2,X_test2,y_test2 = creat_dataset(sz_X,sz_y,seq_len=wins,test_size=test_size)

X_train3,y_train3,X_test3,y_test3 = creat_dataset(zq_X,zq_y,seq_len=wins,test_size=test_size)

X_train4,y_train4,X_test4,y_test4 = creat_dataset(dw_X,dw_y,seq_len=wins,test_size=test_size)

X_train5,y_train5,X_test5,y_test5 = creat_dataset(zs_X,zs_y,seq_len=wins,test_size=test_size)

X_train6,y_train6,X_test6,y_test6 = creat_dataset(hz_X,hz_y,seq_len=wins,test_size=test_size)

X_train7,y_train7,X_test7,y_test7 = creat_dataset(zh_X,zh_y,seq_len=wins,test_size=test_size)

X_train8,y_train8,X_test8,y_test8 = creat_dataset(fs_X,fs_y,seq_len=wins,test_size=test_size)

X_train9,y_train9,X_test9,y_test9 = creat_dataset(jm_X,jm_y,seq_len=wins,test_size=test_size)

X_train = np.concatenate([X_train1,X_train2,X_train3,X_train4,X_train5,X_train6,X_train7,X_train8,X_train9],axis=0)
y_train = np.concatenate([y_train1,y_train2,y_train3,y_train4,y_train5,y_train6,y_train7,y_train8,y_train9],axis=0)

X_test = np.concatenate([X_test1,X_test2,X_test3,X_test4,X_test5,X_test6,X_test7,X_test8,X_test9],axis=0)
y_test = np.concatenate([y_test1,y_test2,y_test3,y_test4,y_test5,y_test6,y_test7,y_test8,y_test9],axis=0)

def creat_batch_dataset(X,y,train=True,buffer_size=1000,batch_size=batch_size):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)

    
train_batch_dataset = creat_batch_dataset(X_train,y_train)
test_batch_dataset = creat_batch_dataset(X_test,y_test,train=False)

checkpoint_file1 = 'best_LSTM_model_cs3.hdf5'
checkpoint_callback1 = ModelCheckpoint(filepath=checkpoint_file1,
                                     monitor='val_loss',
                                     mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)
model = Sequential([
    layers.LSTM(units=128,return_sequences=True,input_shape=X_train1.shape[-2:]),
    layers.Dropout(0.2),
    layers.LSTM(units=64,return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(units=8),
    layers.Dense(1)
])
model.compile(loss='mean_squared_error',optimizer='adam')
print(model.summary())

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
with strategy.scope():
    history1 = model.fit(train_batch_dataset,
                          epochs=20,
                          verbose=1,
                          validation_data=test_batch_dataset,
                          callbacks=[checkpoint_callback1])
    
model.load_weights(checkpoint_file1)
pred = model.predict(X_test)
print("rmse:{}".format(np.sqrt(mse(y_test,pred))))
print("mae:{}".format(mae(y_test,pred)))
