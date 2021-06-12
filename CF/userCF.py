import random

import math
from operator import itemgetter
import pandas as pd
from tqdm import tqdm


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 100
        self.n_rec_movie = [5, 10, 15, 20, 30, 50]

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = ' + str(self.n_rec_movie))

    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.2):
        trainSet_len = 0
        testSet_len = 0
        data_df = pd.read_csv(filename, sep="::", engine='python', names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
        watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
        test_df = pd.concat([
            data_df[data_df.UserId == i].iloc[int((1 - pivot) * watch_count[i]):] for i in
            tqdm(watch_count.index)],
            axis=0)
        test_df = test_df.reset_index()
        train_df = data_df.drop(labels=test_df['index'])
        train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        train = train_df[['UserId', 'MovieId', 'Rating']].values
        test = test_df[['UserId', 'MovieId', 'Rating']].values
        for line in train:
            user, movie, rating = line
            self.trainSet.setdefault(user.astype('int'), {})
            self.trainSet[user.astype('int')][movie.astype('int')] = rating
        for line in test:
            user, movie, rating = line
            self.testSet.setdefault(user.astype('int'), {})
            self.testSet[user.astype('int')][movie.astype('int')] = rating
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')

        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user, N):
        K = self.n_sim_user
        rank = {}
        watched_movies = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        pre_list = []
        recall_list = []
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()
        for N in self.n_rec_movie:
            for i, user, in enumerate(self.trainSet):
                test_movies = self.testSet.get(user, {})
                rec_movies = self.recommend(user, N)
                for movie, w in rec_movies:
                    if movie in test_movies:
                        hit += 1
                    all_rec_movies.add(movie)
                rec_count += N
                test_count += len(test_movies)

            precision = hit / (1.0 * rec_count)
            recall = hit / (1.0 * test_count)
            coverage = len(all_rec_movies) / (1.0 * self.movie_count)
            pre_list.append(precision)
            recall_list.append(recall)
            print('N:%d\tprecisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (N, precision, recall, coverage))


if __name__ == '__main__':
    rating_file = '../data/ml-1m/ratings.dat'
    userCF = UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
