import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score

from scipy import stats
import seaborn as sns
from copy import deepcopy

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# 1.查看数据的样子
train_shape = train.shape
train_describe = train.describe()
# 可以看到数据已经经过了缩放

# 查看缺失值
isnull = pd.isnull(train).values.any()
null_sum = train.isnull().sum()
# 发现没有缺失值

# 查看信息
train.info()

# 在这里float64是连续特征。object是离散特征,int64是id
cat_features = list(train.select_dtypes(include=['object']).columns)

con_features = [con for con in list(train.select_dtypes(include=['float64']).columns) if con not in ['loss', 'id']]

id_col = list(train.select_dtypes(include=['int64']).columns)

# 类别值中属性的个数
cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

uniq_values_in_category = pd.DataFrame.from_items([('cat_name', cat_features),
                                                   ('unique_values', cat_uniques)])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 5)
ax1.hist(uniq_values_in_category.unique_values, bins=50)
ax1.set_title('different class number of different features')
ax1.set_xlabel('class amount')
ax1.set_ylabel('features amount')
ax1.annotate('a feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))

ax2.set_xlim(2, 30)
ax2.set_title('zooming')
ax2.set_xlabel('distinct values in a feature')
ax2.set_ylabel('features')
ax2.grid(True)
ax2.hist(uniq_values_in_category[uniq_values_in_category.unique_values <= 30].unique_values, bins=30)
ax2.annotate('binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))

plt.show()

# 2.查看赔偿值
plt.figure(figsize=(16, 8))
plt.plot(train['id'], train['loss'])
plt.title('loss values per id')
plt.xlabel('id')
plt.ylabel('loss')
plt.legend()
plt.show()

# 这样的分布，使得回归的表现不佳

# 3.计算偏度
# 偏度度量了实值随机变量的均值分布的不对称性
value=stats.mstats.skew(train['loss']).data
# 正态分布的偏度为0，即若数据分布是对称的，偏度为0，一般以1为衡量指标
# 由此可见不是正态分布，数据确实是倾斜的


# 4.对数变换
# 对数据进行对数变换通常可以改变倾斜
log_value=stats.mstats.skew(np.log(train['loss'])).data
# 此时倾斜值可以接受

# 图示对数变换前后
fig,(ax3,ax4)=plt.subplots(1,2)
fig.set_size_inches(16,5)
ax3.hist(train['loss'],bins=50)
ax3.set_title('loss before log')
ax3.grid(True)
ax4.hist(np.log(train['loss']),bins=50,color='g')
ax4.set_title('loss after log')
ax4.grid(True)

plt.show()

# 5.连续值特征
train[con_features].hist(bins=50,figsize=(16,12))
plt.show()

# 6.特征之间的相关性
plt.subplots(figsize=(16,9))
correlation_mat=train[con_features].corr()
sns.heatmap(correlation_mat,annot=True)
plt.show()