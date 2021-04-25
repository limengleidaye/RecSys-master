import random
import pandas as pd
import numpy as np
import time

"""
基于用户的协同过滤,皮尔逊相关系数计算用户相似度,K近邻中K值的选取
"""


class UserBasedCF:
    def __init__(self, filepath):
        self.filepath = filepath
        self.average_rating = {}
        self.testdata_rating = {}
        self.test_len = 0

    #  读取文件
    def read_data(self):
        filepath = self.filepath
        rating_rnames = ['userId', 'movieId', 'rating', 'timestamp']
        rating_data = pd.read_table(filepath + "/ratings.dat", sep='::', header=None, names=rating_rnames,
                                    engine='python')
        self.ratingdata = rating_data.values

    #  划分数据集，训练集占比：proportion
    def split_data(self, proportion=0.8):
        self.train_data = {}
        self.test_data = {}
        for userid, movieid, rating, timestamp in self.ratingdata:
            if random.random() < proportion:
                self.train_data.setdefault(int(userid), {})
                self.train_data[int(userid)][int(movieid)] = float(rating)
            else:
                self.test_data.setdefault(int(userid), {})
                self.test_data[int(userid)][int(movieid)] = float(rating)
                #  为计算MAE、RMSE做准备
                self.testdata_rating.setdefault(int(userid), {})
                self.testdata_rating[int(userid)][int(movieid)] = 0
                self.test_len += 1

    #  基于皮尔逊相关系数的用户相似度计算
    def pearsonSimilarity(self, a, b):
        train = self.train_data
        m = 0
        #  计算分子
        for i in train[a].keys():
            if i in train[b].keys():
                m += (train[a][i] - self.average_rating[a]) * (train[b][i] - self.average_rating[b])
        #  计算分母，分母有争议
        n = np.sqrt(sum(pow(train[a][i] - self.average_rating[a], 2) for i in train[a])) * \
            np.sqrt(sum(pow(train[b][i] - self.average_rating[b], 2) for i in train[b]))
        if m != 0:
            return m / n
        else:
            return 0

    #  构建用户相似度矩阵
    def build_user_similarity_matrix(self):
        train = self.train_data
        # 计算每个用户训练集的评分平均值:
        for user, movies in train.items():
            sum = 0
            num = 0
            for movie, rating in movies.items():
                sum += rating
                num += 1
            self.average_rating[user] = sum / num
        self.user_sim_matrix = dict()
        # 物品用户倒排表
        item_users = dict()
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        # 构建相似度矩阵
        for item, users in item_users.items():
            for u in users:
                self.user_sim_matrix.setdefault(u, dict())
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix[u].setdefault(v, self.pearsonSimilarity(u, v))


    # 给用户推荐K个与之相似的用户喜欢的物品
    def recommend(self, user, k=40, nitem=10):
        train = self.train_data
        rank = dict()
        rated_items = train.get(user, {})
        for neb, sim in sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[:k]:
            for movie, rating in train[neb].items():
                if movie in rated_items:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += sim * rating
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[:nitem])

    #  计算评估指标：准确率、召回率、mae、rmse
    def calculate_evaluation_index(self, k=40, nitem=10):
        train = self.train_data
        test = self.test_data
        #  计算准确率与召回率
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, k=k, nitem=nitem)
            for movie, _ in rank.items():
                if movie in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        #  计算mae、rmse
        MAE = 0
        RMSE = 0
        # 1.计算预测评分
        for user, movies in test.items():
            for movie, rating in movies.items():
                m = 0
                n = 0
                for neb, sim in sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[:k]:
                    if movie in train[neb].keys():
                        n += sim
                        m += sim * (train[neb][movie] - self.average_rating[neb])
                if n != 0:
                    self.testdata_rating[user][movie] = self.average_rating[user] + m / n
                else:
                    self.testdata_rating[user][movie] = self.average_rating[user]
        # 2.计算MAE和RMSE
        for user, movies in test.items():
            for movie, rating in movies.items():
                MAE += abs((rating - self.testdata_rating[user][movie]))
                RMSE += pow((rating - self.testdata_rating[user][movie]), 2)
        MAE = MAE / self.test_len
        RMSE = np.sqrt(RMSE / self.test_len)
        return hit / recall, hit / precision, MAE, RMSE


def test_Ubcfrecommend():
    start = time.time()
    ubcf = UserBasedCF(r"./data/ml-1m")
    ubcf.read_data()
    ubcf.split_data(proportion=0.8)
    ubcf.build_user_similarity_matrix()
    print('                  不同K值下推荐算法的各项指标(准确率、召回率、MAE、RMSE):')
    print("%-3s%20s%20s%20s%20s" % ('K', 'precision', 'recall', 'mae', 'rmse'))
    for k in [20, 40, 60, 80, 100]:
        recall, precision, mae, rmse = ubcf.calculate_evaluation_index(k=k, nitem=5)
        print("%3d%19.2f%%%19.2f%%%19.5f%19.5f" % (k, precision * 100, recall * 100, mae, rmse))
    print(f"基于用户的协同过滤推荐算法花费时间为：{time.time() - start : .6f}秒")


if __name__ == "__main__":
    test_Ubcfrecommend()
