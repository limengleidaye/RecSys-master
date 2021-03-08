import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
import keras
from time import time
# from tensorflow_privacy.privacy.optimizers import dp_optimizer
import pandas as pd
import tensorflow as tf

from model import MF
from utils import create_explicit_ml_1m_dataset

import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)

    plt.figure(figsize=(20, 8), dpi=80)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')

    plt.title("Training and validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(['train_' + metric, 'val_' + metric])
    # plt.xticks(range(1,21))
    plt.savefig('./res/mse.png')
    # plt.show()


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path

    file = '../data/ml-1m/ratings.dat'
    test_size = 0.2
    latent_dim = 15
    # use bias
    use_bias = True
    learning_rate = 0.001
    batch_size = 500
    epochs = 50
    results = []

    output_model_file = os.path.join(".\\res\\model", str(time()))
    callbacks = [keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True), \
                 keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

    # ========================== Create dataset =======================
    feature_columns, train, test = create_explicit_ml_1m_dataset(file, latent_dim, test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = MF(feature_columns, use_bias)
    model.summary()
    # ============================Compile============================
    optimizer = SGD(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer,
                  metrics=['mse'])
    # ==============================Fit==============================
    for i in range(epochs):
        history = model.fit(
            train_X,
            train_y,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1,  # 验证集比例
            # callbacks = callbacks
        )
        pred_y = model.predict(test_X)
        pred_y[pred_y < 1] = 1.
        pred_y[pred_y > 5] = 5.
        _ = np.equal(np.round(pred_y).flatten(), test_y)
        results.append(np.sum(_.astype('int')) / _.size)
    # ===========================Test==============================

    print('test rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))
    plot_metric(history, "loss")
    print(results)
    # pd.DataFrame(history.history).to_csv('./res/log_SGD.csv',index=False)
    # model.save_weights('./res/',overwrite=True)
    # print('export saved model.')

    df=pd.read_csv('../precision.csv')
    df['MF_SGD']=results
    df.to_csv('../precision.csv',index=None)
