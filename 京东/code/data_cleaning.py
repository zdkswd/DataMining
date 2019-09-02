# 8. 数据清洗
import pandas as pd

df_user = pd.read_csv('../data/user_action_sample_join.csv', header=0)
# pd.options.display.float_format = '{:,.3f}'.format  # 输出格式设置，保留三位小数
print(len(df_user))

# 删除无交互记录的用户
df_noaction = df_user[(df_user['addcart_num'].isnull()) & (df_user['buy_num'].isnull()) &
                      df_user['favor_num'].isnull() & df_user['click_num'].isnull()].index
df_user.drop(df_noaction, axis=0, inplace=True)
print(len(df_user))

# 统计并删除无购买记录的用户
# 统计无购买记录的用户
df_zero = df_user[df_user['buy_num'] == 0].index
print(len(df_zero))
# 删除无购买记录用户
df_user.drop(df_zero, axis=0, inplace=True)
df_user

# 删除爬虫及惰性用户
# 认为浏览购买转换比和点击购买转换比小于0.0005的用户为惰性用户
bindex = df_user[df_user['buy_browse_ratio'] < 0.0005].index
print(len(bindex))
df_user.drop(bindex, axis=0, inplace=True)

cindex = df_user[df_user['buy_click_ratio'] < 0.0005].index
print(len(cindex))
df_user.drop(cindex, axis=0, inplace=True)


