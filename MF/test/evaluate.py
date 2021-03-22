import numpy as np
from MF.model import MF
from MF.utils import DataSet
import tensorflow as tf
import pandas as pd
import random

file = '../../data/ml-1m/ratings.dat'
test_size = 0.2
latent_dim = 15
# use bias
use_bias = False
# ========================== Create dataset =======================
dataset = DataSet()
feature_columns, train, test = dataset.create_explicit_ml_1m_dataset(file, latent_dim, test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model=========================================
model = MF(feature_columns, use_bias=use_bias)
model.summary()
# ========================load weights==================================
model.load_weights('../res/my_weights/MF-1.0/')  # with bias(avg+user_bias+item_bias)
p, q, user_bias, item_bias = model.get_layer("mf_layer").get_weights()
# =========================bulid recommend metrix=======================
data_df = dataset.get_dataDf()
recommendation = np.dot(p, q.T)
recommendation = np.add(np.add(np.add(recommendation, ), item_bias.T),
                        np.reshape(data_df, (-1, 1)))  # with bias
rec_df = pd.DataFrame(recommendation.T, index=range(1, recommendation.shape[1] + 1),
                      columns=range(1, recommendation.shape[0] + 1))

# ========================evaluate=================================
'''
对于每个用户 
    对于测试集中该用户已经观看过的每一部电影：
        1) 取样该用户从未观看过的100部电影。假定用户没有看过某部电影等于用户与该电影没有关联关系，即感兴趣程度低；
        2) 推荐系统基于该用户从未观看过的100部电影+测试集中该用户已经观看过的电影，生成按照推荐度从高到低排名的物品列表。取Top-N作为推荐列表
        3) 计算推荐给用户的Top-N推荐的准确性。即如果测试集用户已经观看过的这1部电影出现在TOP-N推荐中，则计1. 最后统计出一个比例
        4) 对每个用的TOP-N推荐的准确性进行整合，形成一个总指标。
'''


def preAndrec(train, test, N):
    hit = 0
    all_pre = 0
    all_rec = 0
    # 统计结果
    all_items = set(data_df['MovieId'].unique())
    for user_id in test.index.unique().values:
        train_items = set(train.loc[user_id].values.flatten())
        test_items = set(test.loc[user_id].values.flatten())
        other_items = all_items - train_items.union(test_items)
        random_items = set(random.sample(other_items, 100))
        random_items = random_items.union(test_items)
        sort_values = rec_df[user_id].loc[random_items].sort_values(ascending=False)[:N]
        hit += len(test_items & set(sort_values.index.values))
        all_pre += N
        all_rec += len(test_items)
    return hit / (all_pre * 1.0), hit / (all_rec * 1.0)


# =========================main===============================
if __name__ == '__main__':
    _, test_df = test_X
    test_index_df = pd.DataFrame(test_df[:, 1], index=test_df[:, 0], columns=['MovieId'])
    test_y_index_df = pd.Series(test_y, index=test_index_df.index)
    test_index_df.drop(index=test_y_index_df[test_y_index_df < 4].index, inplace=True)
    _, train_df = train_X
    train_index_df = pd.DataFrame(train_df[:, 1], index=train_df[:, 0], columns=['MovieId'])
    N = [5, 10, 20, 30, 50]
    precision = []
    recall = []
    for i in N:
        p, r = preAndrec(train_index_df, test_index_df, i)
        precision.append(p)
        recall.append(r)
    print(precision)
    print(recall)
    '''
    pre_dataFrame = pd.read_csv('../../precision.csv', engine='python')
    rec_dataFrame = pd.read_csv('../../recall.csv', engine='python')
    pre_dataFrame['MF_noise-old']=precision
    rec_dataFrame['MF_noise-old']=recall
    pre_dataFrame.to_csv('../../precision.csv',index=False)
    rec_dataFrame.to_csv('../../recall.csv',index=False)'''
