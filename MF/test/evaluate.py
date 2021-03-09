import numpy as np
from MF.model import MF
from MF.utils import create_explicit_ml_1m_dataset
import tensorflow as tf

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
optimizer = tf.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer,
              metrics=['mse'])
# ==============================Fit==============================
history = model.fit(
    train_X,
    train_y,
    epochs=0,
    batch_size=batch_size,
    validation_split=0.1,  # 验证集比例
    # callbacks = callbacks
)
# ========================load weights====================
p, q, _, _ = model.get_layer("mf_layer").get_weights()
recommendation=np.matmul(p, q.T)


# ========================evaluate=================================
def precision(train, test, N):
    hit = 0
    all = 0
    for user in train[:,0]:
        tu = test[user]
        rank = recommendation[user,:].sort()[:N]
        for item in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

#=========================main===============================
if __name__ == '__main__':
    print(precision(train_X,test_X,5))