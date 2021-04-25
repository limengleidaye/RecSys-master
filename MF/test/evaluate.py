import numpy as np
from MF.model import MyModel
from MF.utils import DataSet
import pandas as pd
import random

file = '../../data/ml-1m/ratings.dat'
test_size = 0.2
latent_dim = 15
# use bias
use_bias = False
# ========================== Create dataset =======================
dataset = DataSet(file=file)
feature_columns, train, test, hs = dataset.create_explicit_ml_1m_dataset(latent_dim, test_size, add_noise=True)
train_X, train_y = train
test_X, test_y, _, _ = test
# ============================Build Model=========================================
model = MyModel(feature_columns,hs, use_bias=use_bias)
model.summary()
# ========================load weights==================================
model.load_weights('../res/my_weights/MF-LR-1.0/')  # with bias(avg+user_bias+item_bias)
# p, q, user_bias, item_bias = model.get_layer("mf_layer").get_weights()
# =========================bulid recommend metrix=======================
data_df = dataset.get_dataDf()
num_users, num_items = feature_columns[0]['feat_num'], feature_columns[1]['feat_num']
rec_df = pd.DataFrame(np.zeros(shape=(num_users, num_items))).stack().reset_index()
rec_df.columns = ['UserId', 'MovieId', 'Rating']
rec_df['UserId'] += 1
rec_df['MovieId'] += 1

rec_df['Rating'] = model.predict(rec_df[['UserId', 'MovieId']].values, batch_size=500)
# =============加平均值=======================================

'''
recommendation += data_df[['UserId', 'user_avg_score']].drop_duplicates().set_index('UserId').values
recommendation += data_df[['MovieId', 'item_avg_score']].drop_duplicates().set_index('MovieId').sort_index().reindex(
    index=range(1, data_df['MovieId'].max() + 1), fill_value=0).values.flatten()

rec_df = pd.DataFrame(recommendation.T, index=range(1, recommendation.shape[1] + 1),
                      columns=range(1, recommendation.shape[0] + 1))'''

# ========================evaluate=================================
'''
对于每个用户 
    对于测试集中该用户已经观看过的每一部电影：
        1) 取样该用户从未观看过的1000部电影。假定用户没有看过某部电影等于用户与该电影没有关联关系，即感兴趣程度低；
        2) 推荐系统基于该用户从未观看过的1000部电影+测试集中该用户已经观看过的电影，生成按照推荐度从高到低排名的物品列表。取Top-N作为推荐列表
        3) 计算推荐给用户的Top-N推荐的召回率。即如果测试集用户已经观看过的这1部电影出现在TOP-N推荐中，则计1. 最后统计出一个比例
        4) 对每个用户的TOP-N推荐的召回率进行整合，形成一个总指标。
'''


def eva_rec(train, test, N):
    hit_rec = 0
    all_rec = 0
    # 统计结果
    all_items = set(data_df['MovieId'].unique())
    rec_df.set_index(['UserId','MovieId'])
    for user_id in test.index.unique().values:
        train_items = set(train.loc[user_id].values.flatten())
        test_items = set(test.loc[user_id].values.flatten())
        other_items = all_items - train_items.union(test_items)
        for idx in test_items:
            random_items = random.sample(other_items, 1000)
            random_items.append(idx)
            #=================================获取排序后二级索引中的电影号=========================================
            sort_values = rec_df.loc[(user_id,random_items),:].sort_values(ascending=False)[:N].index.get_level_values(1)
            hit_rec += int(idx in set(sort_values.values))

        all_rec += len(test_items)
    return hit_rec / (all_rec * 1.0)


def eva_acc(train, test, N):
    hit_acc = 0
    all_acc = 0
    # 统计结果
    all_items = set(data_df['MovieId'].unique())
    rec_df.set_index(['UserId', 'MovieId'])
    for user_id in test.index.unique().values:
        train_items = set(train.loc[user_id].values.flatten())
        test_items = set(test.loc[user_id].values.flatten())
        other_items = all_items - train_items.union(test_items)
        random_items = random.sample(other_items, 1000)
        random_items += list(test_items)
        # =================================获取排序后二级索引中的电影号=========================================
        sort_values = rec_df.loc[(user_id, random_items), :].sort_values(ascending=False)[:N].index.get_level_values(1)
        hit_acc += len(set(sort_values.index) & test_items)
        all_acc += N
    return hit_acc / (all_acc * 1.0)


# =========================main===============================
if __name__ == '__main__':
    test_index_df = pd.DataFrame(test_X, columns=['UserId', 'MovieId'])
    test_y_index_df = pd.Series(test_y, index=test_index_df.index)
    test_index_df.drop(index=test_y_index_df[test_y_index_df < 5].index, inplace=True)
    test_index_df = test_index_df.set_index('UserId')

    _, train_df = train_X
    train_index_df = pd.DataFrame(train_df[:, 1], index=train_df[:, 0], columns=['MovieId'])

    N = [5, 10, 20, 30, 50]
    precision = []
    recall = []
    for i in N:
        r = eva_rec(train_index_df, test_index_df, i)
        p = eva_acc(train_index_df, test_index_df, i)
        precision.append(p)
        recall.append(r)
        print(p)
        print(r)

    # ===========================save===============================
'''
    pre_dataFrame = pd.read_csv('../../precision.csv', engine='python')
    rec_dataFrame = pd.read_csv('../../recall.csv', engine='python')
    pre_dataFrame['MF_noise']=precision
    rec_dataFrame['MF_noise']=recall
    pre_dataFrame.to_csv('../../precision.csv',index=False)
    rec_dataFrame.to_csv('../../recall.csv',index=False)'''
