import pandas as pd
import numpy as np
from tqdm import tqdm


def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


def create_explicit_ml_1m_dataset(file, latent_dim=4, test_size=0.2):
    np.random.seed(103)
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
    data_df['avg_score'] = data_df.groupby(by='UserId')['Rating'].transform('mean')  # 用户平均得分

    # 两个隐语义模型的矩阵描述
    user_num, item_num = data_df['UserId'].max()+1, data_df['MovieId'].max()+1
    feature_columns = [[denseFeature('avg_score')],
                       [sparseFeature('user_id', user_num, latent_dim),
                        sparseFeature('item_id', item_num, latent_dim)]]
    # 划分训练集和测试集
    watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')  # 用户观看电影次数
    # print("watch_count:",watch_count)

    test_df = pd.concat([
        data_df[data_df.UserId == i].iloc[int((1 - test_size) * watch_count[i]):] for i in tqdm(watch_count.index)],
        axis=0)
    # print("test_df",test_df)
    test_df = test_df.reset_index()
    train_df = data_df.drop(labels=test_df['index'])
    train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=0.9).reset_index(drop=True)
    test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)

    train_X = [train_df['avg_score'].values, train_df[['UserId', 'MovieId']].values]  # 训练集X：平均得分，用户ID，物品ID
    # print("train_X",train_X)
    train_y = train_df['Rating'].values.astype('int32')  # 训练集Y：评分
    # print("train_y",train_y,train_df['Rating'].shape)
    test_X = [test_df['avg_score'].values, test_df[['UserId', 'MovieId']].values]
    test_y = test_df['Rating'].values.astype('int32')
    return feature_columns, (train_X, train_y), (test_X, test_y)
