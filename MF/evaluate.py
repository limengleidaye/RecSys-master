import numpy as np
from MyModel.model import MyModel
from MyModel.utils import DataSet
from tqdm import tqdm
import pandas as pd
import random

from model import MF
from utils import create_explicit_ml_1m_dataset


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


np.random.seed(0)
file = '../data/ml-100k/u.data'
test_size = 0.2

latent_dim = 32
# use bias
use_bias = True
data_df = pd.read_csv(file, sep="	", engine='python',
                      names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
data_df['avg_score'] = data_df.groupby(by='UserId')['Rating'].transform('mean')
# feature columns
user_num, item_num = data_df['UserId'].max() + 1, data_df['MovieId'].max() + 1
feature_columns = [[denseFeature('avg_score')],
                   [sparseFeature('user_id', user_num, latent_dim),
                    sparseFeature('item_id', item_num, latent_dim)]]
# split train dataset and test dataset
data_df['random'] = np.random.random(size=len(data_df))
test_df = data_df[data_df['random'] < test_size]
# watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
# test_df = pd.concat([
#     data_df[data_df.UserId == i].iloc[int(0.8 * watch_count[i]):] for i in tqdm(watch_count.index)], axis=0)
test_df = test_df.reset_index()
train_df = data_df.drop(labels=test_df['index'])
train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)

train_X = [train_df['avg_score'].values, train_df[['UserId', 'MovieId']].values]
train_y = train_df['Rating'].values.astype('int32')
test_X = [test_df['avg_score'].values, test_df[['UserId', 'MovieId']].values]
test_y = test_df['Rating'].values.astype('int32')

model = MF(feature_columns, use_bias=use_bias)
model.compile(metrics=['mae', 'mse'])
model.summary()

model.load_weights('./res/weights/MF/')
print('test mae: %.5f' % model.evaluate(test_X, test_y)[1])
print('test rmse: %.5f' % np.sqrt(model.evaluate(test_X, test_y)[2]))

p, q, user_bias, item_bias = model.get_layer('mf_layer').get_weights()
avg = data_df.groupby('UserId')['Rating'].mean().values
avg = np.expand_dims(np.insert(avg, 0, 0), axis=1)
recommend_matrix = np.dot(p, q.T) + user_bias + item_bias.T + avg


def get_recommend(userId, list, rev):
    ratings = recommend_matrix[userId][list]
    # temp_list = zip(list,ratings)
    return [item[0] for item in sorted(zip(list, ratings), key=lambda x: x[1], reverse=rev)]


def rec(train, test, N):
    item_num = feature_columns[1][1]['feat_num']
    all_items = set(range(1, item_num))
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
            sort_values = get_recommend(user_id, random_items, rev=True)
            for n in N:
                hits[n] += int(idx in sort_values[:n])
        test_count += len(test_items)
    for n in N:
        recall = hits[n] / (1.0 * test_count)
        print('N:%d\trecall=%.6f\t' % (n, recall))


def auc(train, test, Nu):
    item_num = feature_columns[1][1]['feat_num']
    all_items = set(range(1, item_num))

    test_count = len(test)
    AUC = 0
    for user_id in tqdm(test):
        train_items = train[user_id]
        test_items = test[user_id]
        Pu = len(test_items)
        other_items = all_items - train_items.union(test_items)
        random_items = random.sample(other_items, 100)
        random_items = random_items + list(test_items)
        sort_values = get_recommend(user_id, random_items, rev=False)
        rank_sum = 0
        for i in test_items:
            rank_sum += sort_values.index(i) + 1

        auc_u = (rank_sum - Pu * (Pu + 1) / 2) / (Pu * Nu)
        AUC += auc_u
    print("AUC:%.5f\t" % (AUC/test_count))


# =========================main===============================
if __name__ == '__main__':
    trainSet = {}
    testSet = {}
    for line in train_X[1]:
        user, movie = line
        trainSet.setdefault(user, set())
        trainSet[user].add(movie)
    for line in zip(test_X[1], test_y):
        [user, movie], rating = line
        if rating > 4:
            testSet.setdefault(user, set())
            testSet[user].add(movie)
    rec(trainSet, testSet, [5, 10, 15, 20, 25, 30, 35])
    auc(trainSet, testSet, 100)
