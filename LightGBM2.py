import lightgbm as lgbm
from sklearn import metrics
from sklearn import model_selection
import numpy as np
#1构建模型
model = lgbm.LGBMRegressor(
    objective='regression',
    max_depth=5,
    num_leaves=25,
    learning_rate=0.007,
    n_estimators=1000,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=np.random.randint(10e6))
import numpy as np
#2导入数据
from pandas import read_csv
dataset = read_csv('nihe.csv')
values = dataset.values
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
XY= scaler.fit_transform(values)
Featurenum=3
X= XY[:,0:Featurenum]    
Y = XY[:,Featurenum]
n_train_hours1 = 800
n_train_hours2 = 900
trainX = X[:n_train_hours1, :]
trainY =Y[:n_train_hours1]
validX=X[n_train_hours1:n_train_hours2, :]
validY=Y[n_train_hours1:n_train_hours2]
testX = X[n_train_hours2:, :]
testY =Y[n_train_hours2:]
#3构建、拟合、预测
model.fit(
        trainX,
        trainY,
        eval_set=[(trainX, trainY), (validX, validY)],
        eval_names=('fit', 'val'),
        eval_metric='l2',
        early_stopping_rounds=200,
        verbose=False) 
forecasttestY0 = model.predict(testX)
Hangnum=len(forecasttestY0)
forecasttestY0 = np.reshape(forecasttestY0, (Hangnum, 1))
#4反变换
from pandas import concat
inv_yhat =np.concatenate((testX,forecasttestY0), axis=1)
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,Featurenum]