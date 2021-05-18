import pandas as pd
import numpy as np
import time

"""
矩阵分解
"""


class MF:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_data(self):
        filepath = self.filepath
        rating_rnames = ['userId', 'movieId', 'rating', 'timestamp']
        rating_data = pd.read_table(filepath + "/ratings.dat", sep='::', header=None, names=rating_rnames,
                                    engine='python')
        self.ratingdata = rating_data.values
        #  建立用户-索引、物品-索引键值对
        self.user_index = {}
        self.movie_index = {}
        self.data = []
        user_index = 0
        movie_index = 0
        for userid, movieid, rating, timestamp in self.ratingdata:
            if int(userid) not in self.user_index:
                self.user_index[int(userid)] = user_index
                user_index += 1
            if int(movieid) not in self.movie_index:
                self.movie_index[int(movieid)] = movie_index
                movie_index += 1
            self.data.append([self.user_index[int(userid)], self.movie_index[int(movieid)], float(rating)])
        self.user_num = user_index
        self.movie_num = movie_index

        self.rating_matrix_true = np.zeros((user_index, movie_index))
        self.rating_matrix_test = np.zeros((user_index, movie_index))
        self.rating_matrix_predict = np.zeros((user_index, movie_index))
        return user_index, movie_index

    def split_data(self, proportion=0.8):
        data_df = pd.DataFrame(self.data, columns=['UserId', 'MovieId', 'Rating'])
        watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
        test_df = pd.concat([
            data_df[data_df.UserId == i].iloc[int(0.8 * watch_count[i]):] for i in watch_count.index], axis=0)
        train_df = data_df.drop(labels=test_df.index)
        self.train_data = [[int(i[0]), int(i[1]), i[2]] for i in train_df.values]
        self.test_data = [[int(i[0]), int(i[1]), i[2]] for i in test_df.values]
        for d in self.data:
            self.rating_matrix_true[d[0]][d[1]] = d[2]
        for d in self.test_data:
            self.rating_matrix_test[d[0]][d[1]] = d[2]

    # 初始化矩阵
    def initLFM(self, K=13, i=1):
        p = np.random.random((self.user_num, K))
        q = np.random.random((self.movie_num, K))
        p *= i
        q *= i
        return p, q

    # 矩阵分解  K:隐类的格式， steps:迭代次数， alpha：学习率， beta：正则化参数
    # 直接更新整个向量
    def matrix_factorization_1(self, K=13, steps=30, alpha=0.013, beta=0.015, i=1):
        p, q = self.initLFM(K=K, i=i)
        for step in range(steps):
            for data in self.train_data:
                e = data[2] - np.dot(p[data[0]], q[data[1]].T)
                p1 = e * q[data[1]] - beta * p[data[0]]
                q1 = e * p[data[0]] - beta * q[data[1]]
                p[data[0]] += alpha * p1
                q[data[1]] += alpha * q1
        self.predicted_rating_matrix = np.dot(p, q.T)

    # 逐步更新向量里的每一项值
    def matrix_factorization_2(self, K=13, steps=30, alpha=0.013, beta=0.015, i=1):
        p, q = self.initLFM(K=K, i=i)
        for step in range(steps):
            for data in self.train_data:
                e = data[2] - np.dot(p[data[0]], q[data[1]].T)
                for k in range(K):
                    p1 = e * q[data[1]][k] - beta * p[data[0]][k]
                    q1 = e * p[data[0]][k] - beta * q[data[1]][k]
                    p[data[0]][k] += alpha * p1
                    q[data[1]][k] += alpha * q1
        self.predicted_rating_matrix = np.dot(p, q.T)

    def eva(self, k):
        self.k = k
        hit = 0
        test_count = 0
        for i in range(len(self.user_index)):
            for j in range(len(self.movie_index)):
                if self.rating_matrix_true[i][j] == 0:
                    self.predicted_rating_matrix[i][j] = 0
                if self.rating_matrix_test[i][j] > 0:
                    test_count += 1
        for i in range(len(self.user_index)):
            max_k = sorted(self.predicted_rating_matrix[i], reverse=True)[self.k]
            for j in range(len(self.movie_index)):
                if self.predicted_rating_matrix[i][j] <= max_k:
                    self.predicted_rating_matrix[i][j] = 0
        for i in range(len(self.user_index)):
            for j in range(len(self.movie_index)):
                if self.predicted_rating_matrix[i][j] != 0 and self.rating_matrix_test[i][j] != 0:
                    hit += 1
        precision = hit / (1.0 * self.k * len(self.user_index))
        recall = hit / (1.0 * test_count)
        # print('precision:', precision)
        # print('recall:', recall)
        return precision, recall

    def calculate_evaluation_index(self):
        MAE = 0
        RMSE = 0
        count = len(self.test_data)
        for data in self.test_data:
            predicted_rating = self.predicted_rating_matrix[data[0]][data[1]]
            MAE += abs(data[2] - predicted_rating)
            RMSE += pow((data[2] - predicted_rating), 2)
        MAE = MAE / count
        RMSE = np.sqrt(RMSE / count)
        return MAE, RMSE


def testMF():
    mf = MF(r"../data/ml-1m")
    mf.read_data()
    mf.split_data(proportion=0.8)
    print('                 不同迭代次数下的各项指标(准确率、召回率、MAE、RMSE):')
    print("%1s%20s%20s%20s%20s%20s" % ('n', 'precision', 'recall', 'mae', 'rmse', 'time'))
    n_set = [5, 10, 15, 20, 25, 30]
    for n in n_set:
        start = time.time()
        mf.matrix_factorization_1(K=13, steps=n, alpha=0.013, beta=0.015, i=1)
        mae, rmse = mf.calculate_evaluation_index()
        precision, recall = mf.eva(k=50)
        print("%-19.5f%-20.5f%-19.5f%-20.5f%-10.5f" % (precision, recall, mae, rmse, time.time() - start))


if __name__ == "__main__":
    testMF()
