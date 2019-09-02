# 1.数据集验证
# 首先检查JData_User中用户和JData_Action中用户是否一致
# 保证行为数据中所产生的行为均由用户数据中的用户产生（但是可能存在用户在行为数据中无行为）
# 思路：利用pd.Merge连接sku和Action中的sku，观察Action中的数据是否减少Example

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

'''
df1 = pd.DataFrame({'sku': ['a', 'a', 'e', 'c'], 'data': [1, 1, 2, 3]})
df2 = pd.DataFrame({'sku': ['a', 'b', 'f']})
df3 = pd.DataFrame({'sku': ['a', 'b', 'd']})
df4 = pd.DataFrame({'sku': ['a', 'b', 'c', 'd']})
print(pd.merge(df2, df1))
'''


def user_action_check():
    df_user = pd.read_csv('../data/JData_User.csv', encoding='gbk')
    df_sku = df_user.loc[:, 'user_id'].to_frame()
    df_month02 = pd.read_csv('../data/JData_Action_201602_sample.csv', encoding='gbk')
    print('is action of feb from user file?', len(df_month02) == len(pd.merge(df_sku, df_month02)))


user_action_check()


# 结论：User数据集中的用户和交互行为数据集中的用户完全一致
# 根据merge前后的数据量对比，能保证Action中的用户ID是User中ID的子集

# 2.检查是否有重复记录
def deduplicate(filepath, newpath):
    df_file = pd.read_csv(filepath, encoding='gbk')
    before = df_file.shape[0]
    df_file.drop_duplicates(inplace=True)
    after = df_file.shape[0]
    n_dup = before - after
    if n_dup != 0:
        df_file.to_csv(newpath, index=None)
        print('重复的数字为' + n_dup)
    else:
        print('no duplicate')


'''
deduplicate('../data/JData_Action_201602.csv', '../data/JData_Action_201602_dedup.csv')
'''

'''
df_month02 = pd.read_csv('../data/JData_Action_201602.csv', encoding='gbk')
is_duplicated = df_month02.duplicated()
df_d = df_month02[is_duplicated]
show_dup_type = df_d.groupby('type').count()
'''

# 发现重复数据大多是由于浏览(1)或者点击(6)


# 3.检查是否存在注册时间在2016，4，15号之后的用户
import pandas as pd

df_user = pd.read_csv('../data/JData_User.csv', encoding='gbk')
df_user['user_reg_tm'] = pd.to_datetime(df_user['user_reg_tm'])
show_time = df_user.loc[df_user.user_reg_tm >= '2016-4-15']

# 由于注册时间是京东系统错误造成的，如果行为数据中没有15号之后的数据，
# 说明这些用户还是正常的用户，不需要删除


# 4.行为数据中user_id为浮点型，进行int类型转换

df_month = pd.read_csv('../data/JData_Action_201602_sample.csv', encoding='gbk')
df_month['user_id'] = df_month['user_id'].apply(lambda x: int(x))
print(df_month['user_id'].dtype)

# 5.年龄区间的处理
df_user = pd.read_csv('../data/JData_User.csv', encoding='gbk')


def transAge(x):
    if x == '15岁以下':
        x = '1'
    elif x == '16-25岁':
        x = '2'
    elif x == '26-35岁':
        x = '3'
    elif x == '36-45岁':
        x = '4'
    elif x == '46-55岁':
        x = '5'
    elif x == '56岁以上':
        x = '6'
    return x


df_user['age'] = df_user['age'].apply(transAge)
age_count = df_user.groupby(df_user['age']).count()
# df_user.to_csv('',index=None)


# 为了能够进行清洗，首先构造了简单用户（user）行为特征和商品（item）
# 行为特征，对应于两张表user_table和item_table

# 6. 构建user_table
from collections import Counter


# 功能函数：对每一个user分组的数据进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1.浏览 2.加购 3.删除 4.购买 5.收藏 6.点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['user_id', 'browse_num', 'addcart_num', 'delcart_num',
                  'buy_num', 'favor_num', 'click_num']]


# 由于用户行为数据量较大，一次性读入可能造成内存错误（memory error）
# ------->可以使用pandas的分块（chunk）读取
# 对action数据进行统计
# 根据自己调节chunk_size大小
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True, encoding='gbk')
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[['user_id', 'type']]
            chunks.append(chunk)
            print('iteration')
        except StopIteration:
            loop = False
            print('Iteration is stopped')
    # 将块拼接成pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # 按user_id分组，对每一组进行统计，as_index表示无索引形式返回数据
    df_ac = df_ac.groupby(['user_id'], as_index=False).apply(add_type_count)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates('user_id')

    return df_ac


# 将各个action数据的统计量进行聚合
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data('../data/JData_Action_201602_sample.csv'))
    df_ac.append(get_from_action_data('../data/JData_Action_201603_sample.csv'))
    df_ac.append(get_from_action_data('../data/JData_Action_201604_sample.csv'))

    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()

    # 构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于1的转化率字段设置为1
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1

    return df_ac


# 从user表中抽取需要的字段
def get_from_jdata_user(df_user):
    # df_user = pd.read_csv('../data/JData_User.csv', header=0, encoding='gbk')
    df_user = df_user[['user_id', 'age', 'sex', 'user_lv_cd']]
    return df_user


user_base = get_from_jdata_user(df_user)
user_behavior = merge_action_data()

# 将用户信息与用户行为进行表连接
user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')

# 将拼成的表进行保存
naindex = user_behavior[user_behavior['browse_num'].isna()].index
user_behavior.drop(naindex, axis=0, inplace=True)
user_behavior.to_csv('../data/user_action_sample_join.csv', index=False)


# 至此user_table已经完成


# 7.构建item_table
def get_from_jdata_product():
    df_item = pd.read_csv('../data/JData_Product.csv', header=0, encoding='gbk')
    return df_item


def add_type_count2(group):
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['sku_id', 'browse_num', 'addcart_num', 'delcart_num',
                  'buy_num', 'favor_num', 'click_num']]


# 对action中的数据进行统计
def get_from_action_data2(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['sku_id', 'type']]
            chunks.append(chunk)
            print('iteration')
        except StopIteration:
            loop = False
            print('Iteration is stopped')

    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac.groupby(['sku_id'], as_index=False).apply(add_type_count2)
    df_ac = df_ac.drop_duplicates('sku_id')

    return df_ac


# 获取评论的商品数据，如果存在某一个商品有两个日期的评论，取最晚的那个
def get_from_jdata_comment():
    df_cmt = pd.read_csv('../data/JData_Comment.csv', header=0)
    df_cmt['dt'] = pd.to_datetime(df_cmt['dt'])
    # find latest comment index
    idx = df_cmt.groupby(['sku_id'])['dt'].transform(max) == df_cmt['dt']
    df_cmt = df_cmt[idx]

    return df_cmt[['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]


def merge_action_data2():
    df_ac = []
    df_ac.append(get_from_action_data2('../data/JData_Action_201602_sample.csv'))
    df_ac.append(get_from_action_data2('../data/JData_Action_201603_sample.csv'))
    df_ac.append(get_from_action_data2('../data/JData_Action_201604_sample.csv'))

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.groupby(['sku_id'], as_index=False).sum()

    # 构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于1的转化率字段设置为1
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1

    return df_ac


item_base = get_from_jdata_product()
item_behavior = merge_action_data2()
item_comment = get_from_jdata_comment()

item_behavior = pd.merge(item_base, item_behavior, on=['sku_id'], how='left')
item_behavior = pd.merge(item_behavior, item_comment, on=['sku_id'], how='left')

naindex = item_behavior[item_behavior['browse_num'].isna()].index
item_behavior.drop(naindex, axis=0, inplace=True)
item_behavior.to_csv('../data/item_sample_table.csv', index=False)

# 至此item_table已完成
