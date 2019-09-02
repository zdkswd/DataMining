import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 9.数据探索

# 9.1 周一到周日各天的购买情况
# 提取购买（type=4）的行为数据
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['user_id', 'sku_id', 'type', 'time']]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('stop')
    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4为购买
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[['user_id', 'sku_id', 'time']]


df_ac = []
df_ac.append(get_from_action_data('../data/JData_Action_201602_sample.csv'))
df_ac.append(get_from_action_data('../data/JData_Action_201603_sample.csv'))
df_ac.append(get_from_action_data('../data/JData_Action_201604_sample.csv'))
df_ac = pd.concat(df_ac, ignore_index=True)

print(df_ac.dtypes)

# 将time类型转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])

# 将time转换为星期几
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)

# 周一到周日每天购买个数
df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['weekday', 'user_num']

# 周一到周日每天购买商品的个数
df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['weekday', 'user_num']

# 周一到周日每天购买记录个数
df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['weekday', 'user_item_num']

# 画出图形看趋势
bar_width = 0.2
opacity = 0.4

plt.bar(df_user['weekday'], df_user['user_num'], bar_width, alpha=opacity,
        color='c', label='user')
plt.bar(df_item['weekday'] + bar_width, df_item['user_num'], bar_width, alpha=opacity,
        color='g', label='item')
plt.bar(df_ui['weekday'] + 2 * bar_width, df_ui['user_item_num'], bar_width, alpha=opacity,
        color='m', label='user_item')

plt.xlabel('weekday')
plt.ylabel('number')
plt.title('A week purchase table')
plt.xticks(df_user['weekday'] + bar_width * 3 / 2, (1, 2, 3, 4, 5, 6, 7))
plt.tight_layout()
plt.legend(prop={'size': 10})
plt.show()

# 9.2 一个月中各天购买量
df_ac = get_from_action_data('../data/JData_Action_201602_sample.csv')

df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)

df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['weekday', 'user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['weekday', 'user_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['weekday', 'user_item_num']


# 画图与上面类似，可以看出月份中随时间的变化

# 9.3 商品类别销售统计
# 周一到周日各商品类别销售情况
# 从行为记录中提取商品类别数据
def get_from_action_data_3(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['cate', 'brand', 'type', 'time']]
            chunks.append(chunk)
            print('iteration')
        except StopIteration:
            loop = False
            print('stop')
    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4为购买
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[['cate', 'brand', 'type', 'time']]


df_ac = []
df_ac.append(get_from_action_data_3('../data/JData_Action_201602_sample.csv'))
df_ac.append(get_from_action_data_3('../data/JData_Action_201603_sample.csv'))
df_ac.append(get_from_action_data_3('../data/JData_Action_201604_sample.csv'))
df_ac = pd.concat(df_ac, ignore_index=True)

# 观察有几个品类的商品
show_cate = df_ac.groupby(df_ac['cate']).count()

# 将time类型转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])

# 将time转换为星期几
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)

df_product = df_ac['brand'].groupby([df_ac['time'], df_ac['cate']]).count()
df_product = df_product.unstack()
df_product.plot(kind='bar', figsize=(14, 10), title='cate purchase table in a week')
plt.show()


# 9.4 每月各类商品销售情况（例如只关注8）
# 画出类似9.1的图，三个柱子分别为2月，3月，4月

# 9.5 查看特定用户对特定商品的轨迹
def spec_ui_action_data(fname, user_id, item_id, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['user_id', 'sku_id', 'type', 'time']]
            chunks.append(chunk)
            print('iteration')
        except StopIteration:
            loop = False
            print('stop')
    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac[(df_ac['user_id'] == user_id) & (df_ac['sku_id'] == item_id)]

    return df_ac


def explore_user_item_via_time(user_id, item_id):
    df_ac = []
    df_ac.append(spec_ui_action_data('../data/JData_Action_201602_sample.csv', user_id, item_id))
    df_ac.append(spec_ui_action_data('../data/JData_Action_201603_sample.csv', user_id, item_id))
    df_ac.append(spec_ui_action_data('../data/JData_Action_201604_sample.csv', user_id, item_id))
    df_ac = pd.concat(df_ac, ignore_index=False)
    print(df_ac.sort_values(by='time'))


explore_user_item_via_time(266079, 138778)
