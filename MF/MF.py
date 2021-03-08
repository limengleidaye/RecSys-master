from pylab import *
import numpy as np
import time


class SGD_MF():
    def __init__(self, filename, k, ratio):
        self.filename = filename
        self.k = k
        self.N = 0
        self.M = 0
        self.ratio = ratio

    def RatingMatrixProcessing(self, train, test, N, M, gamma, D, lambda_1, Step):
        # train: train data
        # test: test data
        # N:the number of user
        # M:the number of item
        # gamma: the learning rata
        # D: the number of latent factor
        # lambda_1: regularization parameter
        # Step: the max iteration
        p = np.random.random((N, D))
        q = np.random.random((M, D))
        rmse = []
        loss = []
        for ste in range(Step):
            los = 0.0
            for data in train:
                u = data[0]
                i = data[1]
                r = data[2]

                e = r - np.dot(p[u], q[i].T)
                # print(np.shape(q[i]))
                p[u] = p[u] + gamma * (e * q[i] - lambda_1 * p[u])
                q[i] = q[i] + gamma * (e * p[u] - lambda_1 * q[i])

                los = los + e ** 2 + lambda_1 * (np.square(p[u]).sum() + np.square(q[i]).sum())
            loss.append(los)
            rms = self.RMSE(p, q, test)
            rmse.append(rms)
        # self.rating_matrix_predict = np.dot(p, q.T)
        mae = self.MAE(p, q, test)
        print('MAE:', mae)
        return loss, rmse, p, q

    def GenerateNoiseMatrix(self, e, d, len):
        n = []
        for j in range(len):
            ex = np.random.exponential(1, d)
            no = np.random.normal(0, 1 / d, (d, d))
            nj = 2 * np.sqrt(d) / e * np.dot(np.sqrt(2 * ex), no)
            n.append(nj)
        return n

    def ItemProfileMatrix(self, p, q, train, test, e, gamma, D, lambda_1, Step):
        # train: train data
        # test: test data
        # N:the number of user
        # M:the number of item
        # gamma: the learning rata
        # D: the number of latent factor
        # lambda_1: regularization parameter
        # Step: the max iteration
        rmse = []
        loss = []
        n = self.GenerateNoiseMatrix(e, D, np.shape(q)[0])
        for ste in range(Step):
            los = 0.0
            for data in train:
                u = data[0]
                i = data[1]
                r = data[2]

                error = r - np.dot(p[u], q[i].T)
                q[i] = q[i] + gamma * (e * p[u] - lambda_1 * q[i])

                los = los + error ** 2 + lambda_1 * (np.square(p[u]).sum() + np.square(q[i]).sum()) + np.random.laplace(loc=0,scale=1)
            loss.append(los)
            rms = self.RMSE(p, q, test)
            rmse.append(rms)
        self.rating_matrix_predict = np.dot(p, q.T)
        mae = self.MAE(p, q, test)
        print('MAE:', mae)
        return loss, rmse, p, q

    def RMSE(self, p, q, test):
        count = len(test)
        sum_rmse = 0.0
        for t in test:
            u = t[0]
            i = t[1]
            r = t[2]
            pr = np.dot(p[u], q[i].T)
            sum_rmse += np.square(r - pr)
        rmse = np.sqrt(sum_rmse / count)
        return rmse

    def MAE(self, p, q, test):
        count = len(test)
        sum_mae = 0.0
        for t in test:
            u = t[0]
            i = t[1]
            r = t[2]
            pr = np.dot(p[u], q[i].T)
            sum_mae += abs(r - pr)
        mae = sum_mae / count
        return mae

    def eva(self):
        hit = 0
        test_count = 0
        for i in range(self.N):
            for j in range(self.M):
                if self.rating_matrix_true[i][j] == 0:
                    self.rating_matrix_predict[i][j] = 0
                if self.rating_matrix_test[i][j] != 0:
                    test_count += 1
        for i in range(self.N):
            max_k = sorted(self.rating_matrix_predict[i], reverse=True)[self.k]  # 升序第k个预测
            for j in range(self.M):
                if self.rating_matrix_predict[i][j] <= max_k:
                    self.rating_matrix_predict[i][j] = 0
        for i in range(self.N):
            for j in range(self.M):
                if self.rating_matrix_predict[i][j] != 0 and self.rating_matrix_test[i][j] != 0:
                    hit += 1
        precision = hit / (1.0 * self.k * self.N)
        recall = hit / (1.0 * test_count)
        print('precision:', precision)
        print('recall:', recall)

    def Load_data(self, filedir, ratio):
        user_set = {}
        item_set = {}
        N = 0  # the number of user
        M = 0  # the number of item
        u_idx = 0
        i_idx = 0
        data = []
        f = open(filedir)
        for line in f.readlines():
            u, i, r, t = line.split('::')
            if int(u) not in user_set:
                user_set[int(u)] = u_idx
                u_idx += 1
            if int(i) not in item_set:
                item_set[int(i)] = i_idx
                i_idx += 1
            data.append([user_set[int(u)], item_set[int(i)], float(r)])
        # print(data)

        f.close()
        N = u_idx
        M = i_idx
        self.N = u_idx
        self.M = i_idx
        self.rating_matrix_true = np.zeros((N, M))
        self.rating_matrix_test = np.zeros((N, M))
        self.rating_matrix_predict = np.zeros((N, M))

        np.random.shuffle(data)
        train = data[0:int(len(data) * ratio)]
        test = data[int(len(data) * ratio):]
        for d in data:
            self.rating_matrix_true[d[0]][d[1]] = d[2]
        # print(shape(self.rating_matrix_true))
        for d in test:
            self.rating_matrix_test[d[0]][d[1]] = d[2]
        # print(shape(self.rating_matrix_test))
        return N, M, train, test

    # ----------------------------SELF TEST----------------------------#

    def evaluate(self):
        dir_data = self.filename
        N, M, train, test = self.Load_data(dir_data, self.ratio)
        gamma = 0.0001
        D = 10
        lambda_1 = 0.1
        Step = 100
        e = 10
        loss, rmse, p, q = self.RatingMatrixProcessing(train, test, N, M, gamma, D, lambda_1, Step)
        print('RMSE1:', rmse[-1], 'loss1:', loss[-1])
        loss, rmse, p, q = self.ItemProfileMatrix(p, q, train, test, e, gamma, D, lambda_1, Step)
        print('RMSE2:', rmse[-1], 'loss2:', loss[-1])

        self.eva()


if __name__ == '__main__':
    start = time.time()
    t = SGD_MF("../data/ml-1m/ratings.dat", 5, 0.8)
    t.evaluate()
    print(f"花费时间：{time.time() - start:.6f}")
