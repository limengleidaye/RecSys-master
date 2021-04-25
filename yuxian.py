# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:02:47 2021

@author: nhkj
"""
import random
import pandas as pd
import numpy as np
import time

"基于LSH-MF的推荐系统的K值的选取"


class LshbasedUBCF:
    def __init__(self, filepath):
        self.filepath = filepath
        self.average_rating = {}
        self.traindata_rating = {}
        self.testdata_rating = {}
        self.user_num = 0
        self.movie_num = 0
        self.test_len = 0
        # self.data = []
        self.train_datalsh = {}

    def read_data(self):
        filepath = self.filepath
        rating_rnames = ['userId', 'movieId', 'rating', 'timestamp']
        rating_data = pd.read_table(filepath + "/ratings.dat", sep='::', header=None, names=rating_rnames,
                                    dtype=object, engine='python')
        self.ratingdata = rating_data.values

    def split_data(self):
        self.train_data = {}
        self.test_data = {}
        self.train_movie = set()
        for userid, movieid, rating, timestamp in self.ratingdata:
            if random.random() < 0.8:
                self.train_movie.add(int(movieid))
                self.train_data.setdefault(int(userid), {})
                self.train_data[int(userid)][int(movieid)] = float(rating)
                self.traindata_rating.setdefault(int(userid), {})
                self.traindata_rating[int(userid)][int(movieid)] = 0
            else:
                self.test_data.setdefault(int(userid), {})
                self.test_data[int(userid)][int(movieid)] = float(rating)
                self.testdata_rating.setdefault(int(userid), {})
                self.testdata_rating[int(userid)][int(movieid)] = 0
                self.test_len += 1
        self.train_user = set(self.train_data.keys())
        self.user_num = len(self.train_user)
        self.movie_num = len(self.train_movie)

    def build_rating_matrix(self):
        self.user_index = {}
        self.movie_index = {}
        user_index = 0
        movie_index = 0
        for user in self.train_user:
            self.user_index.setdefault(user, user_index)
            user_index += 1
        for movie in self.train_movie:
            self.movie_index.setdefault(movie, movie_index)
            movie_index += 1
        self.index_user = {v: k for k, v in self.user_index.items()}
        self.index_movie = {v: k for k, v in self.movie_index.items()}
        self.rating_matrix = np.zeros((self.user_num, self.movie_num))
        for user, movies in self.train_data.items():
            for movie, rating in movies.items():
                self.rating_matrix[self.user_index[user]][self.movie_index[movie]] = rating

    def buildHashtable(self, hashtable_num=4, hashfunction_num=4):
        self.hashtable_num = hashtable_num
        self.hashfunction_num = hashfunction_num
        # 生成随机向量
        self.random_matrixes = np.random.uniform(-1, 1, (self.hashtable_num, self.hashfunction_num, self.movie_num))
        # print(self.random_matrixes)
        # 初始化哈希表
        self.hashtables = [{} for i in range(self.hashtable_num)]
        # print(self.hashtables)
        # 每一个用户
        for i, user_vec in enumerate(self.rating_matrix):
            # print("当前用户向量：", user_vec)
            # 每一个hash表
            for j in range(self.hashtable_num):
                # 每次构建随机矩阵
                hashfuc_family = self.random_matrixes[j]
                # print('哈希函数矩阵:',hashfuc_family)
                index = ""
                for k in range(self.hashfunction_num):
                    # index += '1' if user_vec.dot(hashfuc_family[k]) > 0 else '0'
                    r = random.randint(0, 9)
                    if r == 0 and user_vec.dot(hashfuc_family[k]) <= 0:
                        index += '1'
                    elif r != 0 and user_vec.dot(hashfuc_family[k]) > 0:
                        index += '1'
                    else:
                        index += '0'
                # 存入哈希表
                # 桶号转换为2进制
                index = int(index, 2)
                self.hashtables[j].setdefault(index, set())
                self.hashtables[j][index].add(self.index_user[i])

    def findNeighbor(self, user):
        user_vec = self.rating_matrix[self.user_index[user]]
        # 初始化近邻集合
        neighbor_users = set()
        for i in range(self.hashtable_num):
            hashfuc = self.random_matrixes[i]
            index = ''
            for j in range(self.hashfunction_num):
                # index += '1' if user_vec.dot(hashfuc[j]) > 0 else '0'
                # print(index)
                r = random.randint(0, 9)
                if r == 0 and user_vec.dot(hashfuc[j]) <= 0:
                    index += '1'
                elif r != 0 and user_vec.dot(hashfuc[j]) > 0:
                    index += '1'
                else:
                    index += '0'
            index = int(index, 2)
            # x = self.hashtables[index]
            # print(x)
            # |=  对变量值与表达式值执行按位“或”操作，并将结果赋给该变量
            neighbor_users |= self.hashtables[i][index]
            # print(neighbor_users)
        return neighbor_users

    '''def pearsonSimilarity(self, a, b):
        train = self.train_data
        m = 0
        for i in train[a].keys():
            if i in train[b].keys():
                m += (train[a][i] - self.average_rating[a]) * (train[b][i] - self.average_rating[b])
        n = np.sqrt(sum(pow(train[a][i] - self.average_rating[a], 2) for i in train[a])) * \
            np.sqrt(sum(pow(train[b][i] - self.average_rating[b], 2) for i in train[b]))
        if m != 0:
            return m / n
        else:
            return 0'''

    def build_user_similarity_matrix(self):
        train = self.train_data
        # 计算每个用户训练集的评分平均值:
        # self.data = np.zeros((self.user_num, self.movie_num))
        for user, movies in train.items():
            sum = 0
            num = 0
            for movie, rating in movies.items():
                sum += rating
                num += 1
            self.average_rating[user] = sum / num
        # 物品用户倒排表
        item_users = dict()
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        C = {}
        for item, users in item_users.items():
            for u in users:
                C.setdefault(u, {})
                for v in users:
                    if u == v:
                        continue
                    C[u].setdefault(v, 0)
                    C[u][v] += 1
        # 用户相似度矩阵
        self.user_sim_matrix = dict()
        '''for user in self.train_user:
            self.user_sim_matrix.setdefault(user, {})
            for hashneb in self.findNeighbor(user):
                if hashneb == user:
                    continue
                self.user_sim_matrix[user].setdefault(hashneb, self.pearsonSimilarity(hashneb, user))'''
        for u, related_users in C.items():
            self.user_sim_matrix.setdefault(u, {})
            for hashneb in self.findNeighbor(u):
                if hashneb == u:
                    continue
                for v, cuv in related_users.items():
                    self.user_sim_matrix[u].setdefault(hashneb, cuv / np.sqrt(len(train[u]) * len(train[v])))
        for user, movies in train.items():
            for movie, rating in movies.items():
                m = 0
                n = 0
                for neb, sim in sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[0:40]:
                    if movie in train[neb].keys():
                        n += sim
                        m += sim * (train[neb][movie] - self.average_rating[neb])
                if n != 0:
                    self.traindata_rating[user][movie] = self.average_rating[user] + m / n
                else:
                    self.traindata_rating[user][movie] = self.average_rating[user]

    def build_datarating_matrix(self):
        self.user_index = {}
        self.movie_index = {}
        self.data = []
        user_index = 0
        movie_index = 0
        for user in self.train_user:
            self.user_index.setdefault(user, user_index)
            user_index += 1
        for movie in self.train_movie:
            self.movie_index.setdefault(movie, movie_index)
            movie_index += 1
        for user, movies in self.train_data.items():
            for movie, rating in movies.items():
                self.data.append([self.user_index[int(user)], self.movie_index[int(movie)], self.traindata_rating[user][movie]])

    def initLFM(self, K=13, i=1):
        np.random.shuffle(self.data)
        self.traindata_rating1 = self.data[0:int(len(self.data) * 0.8)]
        self.test_data1 = self.data[int(len(self.data) * 0.8):]

        p = np.random.random((self.user_num, K))
        q = np.random.random((self.movie_num, K))
        p *= i
        q *= i
        return p, q

    # 矩阵分解  K:隐类的格式， steps:迭代次数， alpha：学习率， beta：正则化参数
    # 直接更新整个向量
    def matrix_factorization_1(self, K=13, steps=100, alpha=0.013, beta=0.015, i=1):
        p, q = self.initLFM(K=K, i=i)
        for step in range(steps):
            for data in self.traindata_rating1:
                # data = int(data)
                # data = data.astype(np.int32)
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

    def calculate_evaluation_index(self):
        MAE = 0
        RMSE = 0
        count = len(self.test_data1)
        for data in self.test_data1:
            # data = data.type(np.int32)
            predicted_rating = self.predicted_rating_matrix[data[0]][data[1]]
            MAE += abs(data[2] - predicted_rating)
            RMSE += pow((data[2] - predicted_rating), 2)
        MAE = MAE / count
        RMSE = np.sqrt(RMSE / count)
        return MAE, RMSE


def test_LshbasefUbcf():
    lsh_ubcf = LshbasedUBCF(r"./data/ml-1m")
    lsh_ubcf.read_data()
    lsh_ubcf.split_data()
    lsh_ubcf.build_rating_matrix()
    lsh_ubcf.buildHashtable(hashtable_num=3, hashfunction_num=7)
    start = time.time()
    lsh_ubcf.build_user_similarity_matrix()
    print(f"构建用户相似度矩阵花费时间为：{time.time() - start : .6f}秒")
    lsh_ubcf.build_datarating_matrix()
    # lsh_ubcf.split_data(proportion=0.8)
    print('                  不同K值下推荐算法的各项指标(准确率、召回率、MAE、RMSE):')
    '''print("%3s%20s%20s%19s%19s" % ('K', 'precision', 'recall', 'mae', 'rmse'))

    for k in [20, 40, 60, 80, 100]:
    #for k in [20, 40, 60]:
        recall, precision, mae, rmse = lsh_ubcf.calculate_evaluation_index(k=k, nitem=5)
        print("%3d%19.2f%%%19.2f%%%19.5f%19.5f" % (k, precision * 100, recall * 100, mae, rmse))
    print(f"基于局部敏感哈希的协同过滤推荐算法花费时间为：{time.time() - start : .6f}秒")'''
    print("%1s%20s%20s%20s" % ('n', 'mae', 'rmse', 'time'))
    # 参数测试，包括隐类的格式K、迭代步数steps、学习率alpha、正则化参数beta、初始矩阵的构建i
    n_set = [5, 10, 15, 20, 25, 30]
    for n in n_set:
        start = time.time()
        lsh_ubcf.matrix_factorization_1(K=13, steps=n, alpha=0.013, beta=0.015, i=1)
        mae, rmse = lsh_ubcf.calculate_evaluation_index()
        print("%-18d%-19.5f%-20.5f%-10.5f" % (n, mae, rmse, time.time() - start))


if __name__ == "__main__":
    test_LshbasefUbcf()
