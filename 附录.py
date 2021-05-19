import pandas as pd
print(pd.Categorical(['a','b','c','a','e','c']).codes)

#另一种常用于统计建模或机器学习的转换方式是：将分类变量（categorical variable）转换为“哑变量矩阵”（dummy matrix）或“指标矩阵”
# （indicator matrix）。如果DataFrame的某一列中含有k个不同的值，则可以派生出一个k列矩阵或DataFrame（其值全为1和0）。
# pandas有一个get_dummies函数可以实现该功能（其实自己动手做一个也不难）。拿之前的一个例子来说：

df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],'data1': range(6)})
print(df)

print(pd.get_dummies(df['key']))

# 离散特征的编码分为两种情况：
# 1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
# 2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
# 使用pandas可以很方便的对离散型特征进行one-hot编码

import pandas as pd
df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'prize', 'class label']

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
print('df[size]beforeMap:  \n',df['size'] )
df['size'] = df['size'].map(size_mapping)
print('df[size]afterMap:  \n',df['size'] )

class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}
df['class label'] = df['class label'].map(class_mapping)

#Using the get_dummies will create a new column for every unique string in a certain column:
print(pd.get_dummies(df))#通过增加列来增加实现实现one-hot的形式