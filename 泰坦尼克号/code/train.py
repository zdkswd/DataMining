import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('../data/train.csv')

# 1.观察数据
temp = data.head()

# 2.看一下缺失值的状况
null_sum = data.isnull().sum()

# 3.看看整体数据规模
# 只有数值型的数据被展示出来
describe = data.describe()

# 4.查看标签是否均衡
f_label, ax_label = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax_label[0], shadow=True)
ax_label[0].set_title('Survived')
ax_label[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax_label[1])
ax_label[1].set_title('Survived')
plt.show()

# 5.数据分为连续值与离散值
# 如性别就是离散值
sex = data.groupby(['Sex', 'Survived'])['Survived'].count()

f2, ax_sex = plt.subplots(1, 2, figsize=(18, 8))
data[['Survived', 'Sex']].groupby(['Sex']).mean().plot.bar(ax=ax_sex[0])
ax_sex[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax_sex[1])
ax_sex[1].set_title('Sex:Survived vs Dead')
plt.show()

# 6.船舱等级与获救情况，小表格显示
pclass = pd.crosstab(data.Pclass, data.Survived, margins=True)
f3, ax_pclass = plt.subplots(1, 2, figsize=(18, 8))
data['Pclass'].value_counts().plot.bar(ax=ax_pclass[0])
ax_pclass[0].set_title('Number of passagers by Pclass')
ax_pclass[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax_pclass[1])
ax_pclass[1].set_title('Pclass:Survived vs Dead')
plt.show()

# 7.船舱等级与性别 两个因素 对结果的影响，一二三等舱中女性获救比例是否相同
sex_Pclass = pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True)
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
plt.show()

# 8.连续值对结果的影响
# 如年龄
age = data['Age'].describe()
f4, ax_age = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=data, split=True, ax=ax_age[0])
ax_age[0].set_title('Pclass and Age vs Survived')
ax_age[0].set_yticks(range(0, 110, 10))
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, ax=ax_age[1])
ax_age[1].set_title('Sex and Age vs Survived')
ax_age[1].set_yticks(range(0, 110, 10))
plt.show()

# 9.缺失值填充
# 平均值 经验值 回归模型 剔除掉
# 对于年龄可以把人分为几组人，然后将缺失值填充为那组的均值
data['Initial'] = 0
for i in data:
    # 获取每一个人的称呼前缀
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

initial_sex_T = pd.crosstab(data.Initial, data.Sex).T

data['Initial'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady',
                         'Major', 'Master', 'Mlle', 'Mme', 'Rev'], value='others', inplace=True)
initial_avg_age = data.groupby('Initial')['Age'].mean()

# 填充缺失值
data.loc[(data.Age.isnull()) & (data.Initial == 'Miss'), 'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Initial == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial == 'Mrs'), 'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial == 'Ms'), 'Age'] = 28
data.loc[(data.Age.isnull()) & (data.Initial == 'Sir'), 'Age'] = 49
data.loc[(data.Age.isnull()) & (data.Initial == 'others'), 'Age'] = 20

# 查看填充后的结果
after_fill = data.isnull().any()

f5, ax_age_count = plt.subplots(1, 2, figsize=(20, 10))
data[data['Survived'] == 0].Age.plot.hist(ax=ax_age_count[0], bins=20, edgecolor='black', color='red')
ax_age_count[0].set_title('Survived=0')
ax_age_count[0].set_xticks(range(0, 85, 5))
data[data['Survived'] == 1].Age.plot.hist(ax=ax_age_count[1], bins=20, edgecolor='black', color='green')
ax_age_count[1].set_title('Survived=1')
ax_age_count[1].set_xticks(range(0, 85, 5))
plt.show()

sns.factorplot('Pclass', 'Survived', col='Initial', data=data)
plt.show()

# 10.分析登船地点
embarked = pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True)

sns.factorplot('Embarked', 'Survived', data=data)
fig = plt.gcf()
fig.set_size_inches(5, 3)
plt.show()

f6, ax_embarked = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=data, ax=ax_embarked[0, 0])
ax_embarked[0, 0].set_title('Passagers aboard amount')
sns.countplot('Embarked', hue='Sex', data=data, ax=ax_embarked[0, 1])
ax_embarked[0, 1].set_title('Sex split for embarked')
sns.countplot('Embarked', hue='Survived', data=data, ax=ax_embarked[1, 0])
ax_embarked[1, 0].set_title('embarked vs survived')
sns.countplot('Embarked', hue='Pclass', data=data, ax=ax_embarked[1, 1])
ax_embarked[1, 1].set_title('embarked vs pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)
plt.show()

# 码头中也存在缺失值，使用众数来填充
data['Embarked'].fillna('S',inplace=True)
