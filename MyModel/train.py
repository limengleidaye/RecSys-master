import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
import keras
from time import time
# from tensorflow_privacy.privacy.optimizers import dp_optimizer
import pandas as pd
import tensorflow as tf

from model import MyModel
from utils import DataSet

import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path

    file = '../data/ml-1m/ratings.dat'
    test_size = 0.2
    latent_dim = 30
    # use bias
    use_bias = True
    learning_rate = 0.001
    batch_size = 500
    epochs = 50

    output_model_file = os.path.join(".\\res\\callbacks", str(time()))
    callbacks = [keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True), \
                 keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

    # ========================== Create dataset =======================
    feature_columns, train, test, dense_feature = DataSet(file).create_explicit_ml_1m_dataset(latent_dim, test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = MyModel(feature_columns, dense_feature, use_bias=use_bias)
    model.summary()
    # ============================Compile============================
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer,
                  metrics=['mse'])
    # ==============================Fit==============================
    for _ in range(epochs):
        history = model.fit(
            train_X,
            train_y,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1,  # 验证集比例
            # callbacks=callbacks
        )
        # ===========================Test==============================
        print('test rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))

    # p, q, user_bias, item_bias = model.get_layer("mf_layer").get_weights()

    # ===========================Save==============================
    # pd.DataFrame(history.history).to_csv('./res/log/MyModel.csv',index=False)
    model.save_weights('./res/weights/MyModel-LR-1.0/')
    # print('export saved callbacks.')
