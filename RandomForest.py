#今天研究了一下随机森林算法RandomForestClassifier的源码，把一些常用的属性找了出来（针对0/1这种对错判断）

# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree import export_graphviz

final = open('..\\2.svm\\testSet.txt' , 'r')
data = [line.strip().split('\t') for line in final]
feature = [[float(x) for x in row[0:1]] for row in data]
target = [int(row[2]) for row in data]

#拆分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.1, random_state=42)

#分类型决策树
clf = RandomForestClassifier(n_estimators = 8)

#训练模型
s = clf.fit(feature_train , target_train)
print (s)

#评估模型准确率
r = clf.score(feature_test , target_test)
print (r)

print ('判定结果：%s' % clf.predict(feature_test[0]))
#print clf.predict_proba(feature_test[0])

print ('所有的树:%s' % clf.estimators_)

print (clf.classes_)
print (clf.n_classes_)

print ('各feature的重要性：%s' % clf.feature_importances_)

print (clf.n_outputs_)

def _parallel_helper(obj, methodname, *args, **kwargs):
    return getattr(obj, methodname)(*args, **kwargs)

all_proba = Parallel(n_jobs=10, verbose=clf.verbose, backend="threading")(
            delayed(_parallel_helper)(e, 'predict_proba', feature_test[0]) for e in clf.estimators_)
print ('所有树的判定结果：%s' % all_proba)

proba = all_proba[0]
for j in range(1, len(all_proba)):
    proba += all_proba[j]
proba /= len(clf.estimators_)
print ('数的棵树：%s ， 判不作弊的树比例：%s' % (clf.n_estimators , proba[0,0]))
print ('数的棵树：%s ， 判作弊的树比例：%s' % (clf.n_estimators , proba[0,1]))

#当判作弊的树多余不判作弊的树时，最终结果是判作弊
print ('判断结果：%s' % clf.classes_.take(np.argmax(proba, axis=1), axis=0))

#把所有的树都保存到word
for i in range(len(clf.estimators_)):
    export_graphviz(clf.estimators_[i] , '%d.dot'%i)
