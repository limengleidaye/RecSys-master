import numpy as np
from MyModel.model import MyModel
from MyModel.utils import DataSet
from tqdm import tqdm
import pandas as pd
import random

file = '../../data/ml-1m/ratings.dat'
test_size = 0.2
latent_dim = 30
# use bias
use_bias = False
# ========================== Create dataset =======================
dataset = DataSet(file=file)
feature_columns, train, test, dense_feature = dataset.create_explicit_ml_1m_dataset(latent_dim, test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model=========================================
model = MyModel(feature_columns, dense_feature, use_bias=use_bias)
model.compile(metrics=['mae', 'mse'])
model.summary()
# ========================load weights==================================
model.load_weights('../res/weights/MyModel-LR-1m/')  # with bias(avg+user_bias+item_bias)
print('test mae: %.6f'% model.evaluate(test_X, test_y)[1])
print('test rmse: %.6f'% np.sqrt(model.evaluate(test_X, test_y)[2]))
# ========================evaluate=================================
'''
对于每个用户 
    对于测试集中该用户已经观看过的每一部电影：
        1) 取样该用户从未观看过的1000部电影。假定用户没有看过某部电影等于用户与该电影没有关联关系，即感兴趣程度低；
        2) 推荐系统基于该用户从未观看过的1000部电影+测试集中该用户已经观看过的电影，生成按照推荐度从高到低排名的物品列表。取Top-N作为推荐列表
        3) 计算推荐给用户的Top-N推荐的召回率。即如果测试集用户已经观看过的这1部电影出现在TOP-N推荐中，则计1. 最后统计出一个比例
        4) 对每个用户的TOP-N推荐的召回率进行整合，形成一个总指标。
'''


def get_recommend(userId, list):
    temp_list = [[int(userId), int(m_id)] for m_id in list]
    pred_rating = model.predict(temp_list, batch_size=500).flatten()
    return [item[0] for item in sorted(zip(list, pred_rating), key=lambda x: x[1], reverse=True)]


def rec(train, test, N):
    item_num = dataset.get_feature()[1]['feat_num']
    all_items = set(range(1, item_num + 1))
    hits = {}
    for n in N:
        hits.setdefault(n, 0)
    test_count = 0

    # 统计结果
    # user_num = dataset.get_feature()[0]['feat_num']
    for user_id in tqdm(test):
        train_items = train[user_id]
        test_items = test[user_id]
        other_items = all_items - train_items.union(test_items)
        for idx in test_items:
            random_items = random.sample(other_items, 500)
            random_items.append(idx)
            # =================================获取排序后二级索引中的电影号=========================================
            sort_values = get_recommend(user_id, random_items)
            for n in N:
                hits[n] += int(idx in sort_values[:n])
        test_count += len(test_items)
    for n in N:
        recall = hits[n] / (1.0 * test_count)
        print('N:%d\trecall=%.6f\t' % (n, recall))


# =========================main===============================
if __name__ == '__main__':
    trainSet = {}
    testSet = {}
    for line in train_X:
        user, movie = line
        trainSet.setdefault(user, set())
        trainSet[user].add(movie)
    for line in zip(test_X, test_y):
        [user, movie], rating = line
        if rating > 4:
            testSet.setdefault(user, set())
            testSet[user].add(movie)
    rec(trainSet, testSet, [5, 10, 15, 20, 30, 50])
