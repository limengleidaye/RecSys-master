import pandas as pd
import numpy as np
from tqdm import tqdm


class DataSet:
    def __init__(self, file, epsilon=10):
        np.random.seed(0)
        self.data_df = pd.read_csv(file, sep="::", engine='python',
                                   names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
        self.data_len = len(self.data_df)
        self.epsilon = epsilon

    @staticmethod
    def sparseFeature(feat, feat_num, embed_dim=4):
        return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

    @staticmethod
    def denseFeature(feat):
        return {'feat': feat}

    def global_effect(self):
        # 定义隐私预算
        g_epsilon = self.epsilon * 0.02
        i_epsilon = self.epsilon * 0.14
        u_epsilon = self.epsilon * 0.22
        i_beta = 0.1
        u_beta = 0.1
        # =================private global effects===================
        r_max, r_min = self.data_df['Rating'].max(), self.data_df['Rating'].min()
        self._r = r_max - r_min
        GAvg = self.data_df['Rating'].mean() + np.random.laplace(scale=self._r / g_epsilon) / self.data_len

        # ====================加噪声的电影平均得分===========================
        self.data_df['item_avg_score'] = self.data_df.groupby(by='MovieId')['Rating'].transform(
            lambda x: (x.sum() + i_beta * GAvg + np.random.laplace(scale=self._r / i_epsilon)) / (
                    len(x) + i_beta)).clip(
            r_min,
            r_max)
        # =====================加噪声的用户平均得分========================
        self.data_df['R_'] = self.data_df['Rating'] - self.data_df['item_avg_score']
        GAvg_ = self.data_df['R_'].mean() + np.random.laplace(scale=self._r / g_epsilon) / self.data_len
        self.data_df['user_avg_score'] = self.data_df.groupby(by='UserId')['R_'].transform(
            lambda x: (x.sum() + u_beta * GAvg_ + np.random.laplace(scale=self._r / u_epsilon)) / (
                    len(x) + u_beta)).clip(-2,
                                           2)

    def create_explicit_ml_1m_dataset(self, latent_dim=4, test_size=0.2):
        self.global_effect()

        # 两个隐语义模型的矩阵描述
        user_num, item_num = self.data_df['UserId'].max(), self.data_df['MovieId'].max()
        self.feature_columns = [DataSet.sparseFeature('user_id', user_num, latent_dim),
                                DataSet.sparseFeature('item_id', item_num, latent_dim)]

        # 划分训练集和测试集
        watch_count = self.data_df.groupby(by='UserId')['MovieId'].agg('count')  # 用户观看电影次数
        # print("watch_count:",watch_count)

        test_df = pd.concat([
            self.data_df[self.data_df.UserId == i].iloc[int((1 - test_size) * watch_count[i]):] for i in
            tqdm(watch_count.index)],
            axis=0)

        test_df = test_df.reset_index()
        train_df = self.data_df.drop(labels=test_df['index'])
        train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)

        train_X = train_df[['UserId', 'MovieId']].values  # 训练集X：用户ID，物品ID
        # print("train_X",train_X)
        # ==================训练集加噪声===================================
        train_y = train_df[
                      'Rating'].values + np.random.laplace(scale=self._r / (self.epsilon * 0.62),
                                                           size=train_df.shape[0])  # 训练集Y：评分
        test_X = test_df[['UserId', 'MovieId']].values
        test_y = test_df['Rating'].values.astype('int32')

        # ==========================用户、电影平均分，用户、电影最高分=====================================
        item_avg_score = self.data_df[['MovieId', 'item_avg_score']].drop_duplicates().set_index(
            'MovieId').sort_index().reindex(
            index=range(1, item_num + 1), fill_value=0).values.flatten()
        user_avg_score = self.data_df[['UserId', 'user_avg_score']].drop_duplicates().set_index(
            'UserId').sort_index().values.flatten()
        user_highest_score = self.data_df.groupby('UserId')['Rating'].apply(lambda x: x.mode()[0]).values
        movie_highest_score = self.data_df.groupby('MovieId')['Rating'].apply(lambda x: x.mode()[0]).reindex(
            index=range(1, item_num + 1),
            fill_value=0).values

        return self.feature_columns, (train_X, train_y), (test_X, test_y), (
            (user_highest_score, movie_highest_score), (user_avg_score, item_avg_score))

    def get_feature(self):
        return self.feature_columns
