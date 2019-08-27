# 提供的数据为2年内按小时做的自行车租赁数据，
# 其中训练集由每个月的前19天组成，
# 测试集由20号之后的时间组成。

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../data/kaggle_bike_competition_train.csv', header=0, error_bad_lines=False)



# 1.把datetime域切成 日期 和 时间 两部分。
# 处理时间字段
temp = pd.DatetimeIndex(data['datetime'])
data['date'] = temp.date
data['time'] = temp.time
show = data.head()

# 时间那部分，好像最细的粒度也只到小时，
# 所以干脆把小时字段拿出来作为更简洁的特征。
# 设定hour这个小时字段
data['hour'] = pd.to_datetime(data.time, format="%H:%M:%S")
data['hour'] = pd.Index(data['hour']).hour
data



# 2.数据只告诉是哪天，按照一般逻辑，应该周末和工作日出去的人数量不同。
# 我们设定一个新的字段dayofweek表示是一周中的第几天。
# 再设定一个字段dateDays表示离开始租车这个活动多久了，随着时间推移也许会有变化


# 我们对时间类的特征做处理，产出一个星期几的类别型变量
data['dayofweek'] = pd.DatetimeIndex(data.date).dayofweek

# 对时间类特征处理，产出一个时间长度变量
data['dateDays'] = (data.date - data.date[0]).astype('timedelta64[D]')

data



# 3.做一个小小的统计来看看真实的数据分布，
# 统计一下一周各天的自行车租赁情况(分注册的人和没注册的人)

byday = data.groupby('dayofweek')
# 统计下没注册的用户租赁情况
df_casual = byday['casual'].sum().reset_index()

# 统计下注册的用户的租赁情况
df_registered=byday['registered'].sum().reset_index()

# 单独拿一列出来给星期六，再单独拿一列出来给星期日
data['Saturday']=0
data.Saturday[data.dayofweek==5]=1

data['Sunday']=0
data.Sunday[data.dayofweek==6]=1

data




# 4.从数据中，把原始的时间字段等踢掉
# remove old data features
dataRel = data.drop(['datetime', 'count','date','time','dayofweek'], axis=1)



# 5.特征向量化
# 对于pandas的dataframe我们有方法/函数可以直接转成python中的dict。
# 另外，在这里我们要对离散值和连续值特征区分一下了，以便之后分开做不同的特征处理。

from sklearn.feature_extraction import DictVectorizer
# 我们把连续值的属性放入一个dict中
featureConCols = ['temp','atemp','humidity','windspeed','dateDays','hour']
dataFeatureCon = dataRel[featureConCols]
dataFeatureCon = dataFeatureCon.fillna( 'NA' ) #in case I missed any
X_dictCon = dataFeatureCon.T.to_dict().values()

# 把离散值的属性放到另外一个dict中
featureCatCols = ['season','holiday','workingday','weather','Saturday', 'Sunday']
dataFeatureCat = dataRel[featureCatCols]
dataFeatureCat = dataFeatureCat.fillna( 'NA' ) #in case I missed any
X_dictCat = dataFeatureCat.T.to_dict().values()

# 向量化特征
vec = DictVectorizer(sparse = False)
# 此时字典与转换的向量列的顺序没有一一对应
X_vec_cat = vec.fit_transform(X_dictCat)
X_vec_con = vec.fit_transform(X_dictCon)



# 6.标准化连续值特征
# 要对连续值属性做一些处理，最基本的当然是标准化，让连续值属性处理过后均值为0，方差为1。
# 这样的数据放到模型里，对模型训练的收敛和模型的准确性都有好处
from sklearn import preprocessing
# 标准化连续值数据
scaler = preprocessing.StandardScaler().fit(X_vec_con)
X_vec_con = scaler.transform(X_vec_con)



# 7.类别特征编码
# one-hot编码
from sklearn import preprocessing
# one-hot编码
enc = preprocessing.OneHotEncoder()
enc.fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()


# 8.把特征拼一起
# 把离散和连续的特征都组合在一起
import numpy as np
# combine cat & con features
X_vec = np.concatenate((X_vec_con,X_vec_cat), axis=1)


# 9.对结果值也处理一下
# 对Y向量化
Y_vec_reg = dataRel['registered'].values.astype(float)
Y_vec_cas = dataRel['casual'].values.astype(float)