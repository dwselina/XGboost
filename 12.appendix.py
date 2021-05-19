import numpy as np
import pandas as pd
import tensorflow as tf
a=[[1,3],[2,4],[3,5]]
a=np.array(a)
print(a)
print(a.flatten())
# matrix是array的分支，很多情况下matrix和array都是通用的。
# array更灵活，速度更快，很多人把二维的array也翻译成矩阵。
# 但是matrix的优势就是相对简单的运算符号，比如两个矩阵相乘，就是用符号*，但是array相乘不能这么用，得用方法.dot()
# array的优势就是不仅仅表示二维，还能表示3、4、5...维，而且在大部分Python程序里，array也是更常用的。
import numpy as np
a = [[1, 2, 3, ],
        [3, 2, 1]]
type(a)
list
myMat = np.mat(a)
print(myMat )
type(myMat)
x1 = np.arange(9.0).reshape((3, 3))
print(x1)
x2 = np.arange(3.0)
print(x2)
print(np.multiply(x1, x2))


#=====自适应学习率===================
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
path = '8.Advertising.csv'
data = pd.read_csv(path)  # TV、Radio、Newspaper、Sales
x = data[['TV', 'Radio', 'Newspaper']]
# x = lstm-ptb-data[['TV', 'Radio']]
# x = lstm-ptb-data[['TV']]
y = data['Sales']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# print x_train, y_train
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)
print(model)
print(linreg.coef_)
print(linreg.intercept_)

y_hat = linreg.predict(np.array(x_test))
mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print(mse, rmse)



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

mse= tf.reduce_mean(tf.square(y_test-y_hat))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(mse)
learning_rate = tf.placeholder(tf.float32, shape=[])
# ...
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(mse)

sess = tf.Session()

# Feed different values for learning rate to each training step.
result1=sess.run(train_step, feed_dict={learning_rate: 0.1})
result2=sess.run(train_step, feed_dict={learning_rate: 0.1})
result3=sess.run(train_step, feed_dict={learning_rate: 0.01})
result4=sess.run(train_step, feed_dict={learning_rate: 0.01})
for i in range(4):
        print('result%d'%i,result1)

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
# batch = tf.Variable(0)
#
# learning_rate = tf.train.exponential_decay(
#   0.01,                # Base learning rate.
#   batch * BATCH_SIZE,  # Current index into the dataset.
#   train_size,          # Decay step.
#   0.95,                # Decay rate.
#   staircase=True)
# # Use simple momentum for the optimization.
# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)