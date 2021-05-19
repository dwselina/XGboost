import numpy as np
from sklearn.model_selection import train_test_split
# import os
# import csv
# from sklearn.datasets import base
# def load_boston():
#     """Load and return the boston house-prices dataset (regression).
#
#     ==============     ==============
#     Samples total                 506
#     Dimensionality                 13
#     Features           real, positive
#     Targets             real 5. - 50.
#     ==============     ==============
#
#     Returns
#     -------
#     lstm-ptb-data : Bunch
#         Dictionary-like object, the interesting attributes are:
#         'lstm-ptb-data', the lstm-ptb-data to learn, 'target', the regression targets,
#         and 'DESCR', the full description of the dataset.
#
#     Examples
#     --------
#     >>> from sklearn.datasets import load_boston
#     >>> boston = load_boston()
#     >>> print(boston.lstm-ptb-data.shape)
#     (506, 13)
#     """
#     module_path = dirname(__file__)
#
#     fdescr_name = os.path.join(module_path, 'descr', 'boston_house_prices.rst')
#     with open(fdescr_name) as f:  #with用法可以自动关闭，读取文件声明
#         descr_text = f.read()
#
#     data_file_name = os.path.join(module_path, 'lstm-ptb-data', 'boston_house_prices.csv')
#     with open(data_file_name) as f:
#         data_file = csv.reader(f)  #csv读取文件，获得迭代器
#         temp = next(data_file)     #读取文件头，获得总行数和字段名
#         n_samples = int(temp[0])
#         n_features = int(temp[1])
#         lstm-ptb-data = np.empty((n_samples, n_features)) #建立矩阵
#         target = np.empty((n_samples,))
#         temp = next(data_file)  # names of features
#         feature_names = np.array(temp)
#
#         for i, d in enumerate(data_file):  #一行一行通过迭代器读取文件设置到矩阵中
#             lstm-ptb-data[i] = np.asarray(d[:-1], dtype=np.float)
#             target[i] = np.asarray(d[-1], dtype=np.float)
#
#     return Bunch(lstm-ptb-data=lstm-ptb-data,
#                  target=target,
#                  # last column is target value
#                  feature_names=feature_names[:-1],
#                  DESCR=descr_text)



def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]
path = '..\\10.RandomForest\\8.iris.lstm-ptb-data'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
x, y = np.split(data, (4,), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn import ensemble
# 点击率预估模型涉及的训练样本一般是上亿级别，样本量大，模型常采用速度较快的LR。但LR是线性模型，学习能力有限，此时特征工程尤其重要。现有的特征工程实验，
# 主要集中在寻找到有区分度的特征、特征组合，折腾一圈未必会带来效果提升。GBDT算法的特点正好可以用来发掘有区分度的特征、特征组合，
# 减少特征工程中人力成本，且业界现在已有实践，GBDT+LR、GBDT+FM等都是值得尝试的思路。不同场景，GBDT融合LR/FM的思路可能会略有不同，可以多种角度尝试。
clf=ensemble.GradientBoostingClassifier()
gbdt_model=clf.fit(x_train,y_train)
predicty_x=gbdt_model.predict_proba(x_test)[:,1]

print(predicty_x)


