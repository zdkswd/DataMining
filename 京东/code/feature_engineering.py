# 10.特征工程

import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np

test = pd.read_csv('../data/JData_Action_201602_sample.csv')
print(test.dtypes)
print(test.info())
test[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = test[
    ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
print(test.dtypes)
print(test.info())


# 获取某个时间片段的行为记录
def get_actions(start_time, end_time, all_actions):
    actions = all_actions[(all_actions.time >= start_time) & (all_actions.time < end_time)].copy()
    return actions


# 10.1 用户基本特征
# 获取基本用户特征，基于用户本身属性多为类别特征的特点，对age,sex,user_lv_cd进行
# 独热编码操作，对于用户注册时间暂时不处理
from sklearn import preprocessing


def get_basic_user_feat():
    # 针对年龄的中文字符问题处理，变为类别值再独热编码，对于sex也是独热编码
    user = pd.read_csv('../data/JData_User.csv', encoding='gbk')
    user.dropna(axis=0, how='any', inplace=True)

    le = preprocessing.LabelEncoder()
    age_df = le.fit_transform(user['age'])
    age_df = age_df.astype(int)
    age_df = pd.get_dummies(age_df, prefix='age')

    user['sex'] = user['sex'].astype(int)
    sex_df = pd.get_dummies(user['sex'], prefix='sex')

    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    user_lv_df = user_lv_df.astype(int)

    user = pd.concat([user['user_id'], age_df, sex_df], axis=1)
    return user


user = get_basic_user_feat()


# 10.2 商品特征
# 商品基本特征
# 根据商品文件获取基本特征，针对属性a1,a2,a3进行独热编码，商品类别和品牌直接作为特征
def get_basic_product_feat():
    product = pd.read_csv('../data/JData_Product.csv')
    attr1_df = pd.get_dummies(product['a1'], prefix='a1')
    attr2_df = pd.get_dummies(product['a2'], prefix='a2')
    attr3_df = pd.get_dummies(product['a3'], prefix='a3')
    product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
    return product


product = get_basic_product_feat()

# 10.3 评论特征
# 分时间段
# 对评论数进行独热编码
comment_date = ['2016-02-01', '2016-02-08', '2016-02-15', '2016-02-22', '2016-02-29',
                '2016-03-07', '2016-03-14', '2016-03-21', '2016-03-28', '2016-04-04',
                '2016-04-11', '2016-04-15']


def get_comment_product_feat(end_date):
    comments = pd.read_csv('../data/JData_Comment.csv')
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    comments = comments[comments.dt == comment_date_begin]
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0
    df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]

    comments = pd.concat([comments, df], axis=1)

    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0',
                         'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
    return comments
