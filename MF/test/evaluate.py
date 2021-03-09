import numpy as np
from MF.model import MF
from MF.utils import create_explicit_ml_1m_dataset
import tensorflow as tf
import pandas as pd

file = '../../data/ml-1m/ratings.dat'
test_size = 0.2
latent_dim = 15
# use bias
use_bias = True
learning_rate = 0.001
batch_size = 500
epochs = 100
# ========================== Create dataset =======================
feature_columns, train, test = create_explicit_ml_1m_dataset(file, latent_dim, test_size)
train_X, train_y = train
test_X, test_y = test
# ============================Build Model==========================
model = MF(feature_columns, use_bias)
model.summary()
# ============================Compile============================
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer,
              metrics=['mse'])
# ==============================Fit==============================
history = model.fit(
    train_X,
    train_y,
    epochs=0,
    batch_size=batch_size,
    validation_split=0.1,  # 验证集比例
)

model.load_weights('../res/my_weights/')
# print(model.evaluate(test_X,test_y))
# ========================load weights====================
p, q, user_bias, item_bias = model.get_layer("mf_layer").get_weights()
data_df = pd.read_csv(file, sep="::", engine='python', names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
user_avg = data_df.groupby('UserId')['Rating'].mean().values
recommendation = np.matmul(p, q.T)[1:,1:]
recommendation = np.add(np.add(np.add(recommendation, user_bias[1:]),item_bias[1:].T),np.reshape(user_avg,(-1,1))).round()
recommendation[recommendation>5]=5
recommendation[recommendation<1]=1

# ========================evaluate=================================
'''
对于每个用户
    对于测试集中该用户已经观看过的每一部电影：
        1) 取样该用户从未观看过的100部电影。假定用户没有看过某部电影等于用户与该电影没有关联关系，即感兴趣程度低；
        2) 推荐系统基于该用户从未观看过的100部电影+测试集中该用户已经观看过的电影，生成按照推荐度从高到低排名的物品列表。取Top-N作为推荐列表
        3) 计算推荐给用户的Top-N推荐的准确性。即如果测试集用户已经观看过的这1部电影出现在TOP-N推荐中，则计1. 最后统计出一个比例
        4) 对每个用的TOP-N推荐的准确性进行整合，形成一个总指标。
'''
def precision(train, test, N):
    hit = 0
    all = 0

    return hit / (all * 1.0)

# =========================main===============================
# if __name__ == '__main__':
# print(precision(train_X,test_X,5))
