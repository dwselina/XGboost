import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
#1. load dataset
from pandas import read_csv
dataset = read_csv('nihe.csv')
values = dataset.values
#2.tranform lstm-ptb-data to [0,1]  3个属性，第4个是待预测量
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
XY= scaler.fit_transform(values)
Featurenum=3
X= XY[:,0:Featurenum]    
Y = XY[:,Featurenum]
#3.split into train and test sets 950个训练集，剩下的都是验证集
n_train_hours1 = 800
n_train_hours2 = 900
trainX = X[:n_train_hours1, :]
trainY =Y[:n_train_hours1]
validX=X[n_train_hours1:n_train_hours2, :]
validY=Y[n_train_hours1:n_train_hours2]
testX = X[n_train_hours2:, :]
testY =Y[n_train_hours2:]

#3构建、拟合、预测
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
model.fit(trainX, trainY)
forecasttestY0 = model.predict(testX)
Hangnum=len(forecasttestY0)
forecasttestY0 = np.reshape(forecasttestY0, (Hangnum, 1))
plot_importance(model)
plt.show()

#4反变换
from pandas import concat
inv_yhat =np.concatenate((testX,forecasttestY0), axis=1)
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,Featurenum]