import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.config.experimental_run_functions_eagerly(True)
np.random.seed(0)


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Layer
        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(seed=10),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(seed=15),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(seed=20),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='item_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(seed=30),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs):
        user_id, item_id, avg_score = inputs

        # MF
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id-1)  # 选取一个张量里面索引对应的元素
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id-1)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=1, keepdims=True)  # 用户对物品的评分
        # print(tf.shape(outputs))
        # MF-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id-1)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id-1)
        bias = tf.reshape((avg_score + user_bias + item_bias), shape=(-1, 1))
        # use bias
        outputs = bias + outputs if self.use_bias else outputs
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        avg_score = tf.keras.Input(shape=(), dtype=tf.float32)
        tf.keras.Model(inputs=[user_id, item_id, avg_score], outputs=self.call([user_id, item_id, avg_score])).summary()


class Noise(Layer):
    def __init__(self, item_num):
        super(Noise, self).__init__()
        self.item_num = item_num
        self.laplace = [np.random.laplace() for i in range(self.item_num)]

    def build(self, input_shape):
        self.noise = self.add_weight(name='noise',
                                     shape=(self.item_num, 1),
                                     initializer=tf.constant_initializer(self.laplace),
                                     trainable=False
                                     )

    def call(self, inputs):
        output, item_id = inputs
        add_noise = tf.nn.embedding_lookup(params=self.noise, ids=item_id-1)
        output += add_noise
        return output


class MF(tf.keras.Model):
    def __init__(self, feature_columns, implicit=False, use_bias=True, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param implicit: whether implicit or not
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # print(feature_columns)
        num_users, num_items = self.sparse_feature_columns[0]['feat_num'], \
                               self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        # print(num_users,num_items)
        self.mf_layer = MF_layer(num_users, num_items, latent_dim, use_bias,
                                 user_reg, item_reg, user_bias_reg, item_bias_reg)
        self.noise = Noise(num_items)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        avg_score = dense_inputs
        output = self.mf_layer([user_id, item_id, avg_score])  # 前一层的输出，前一层的预测
        #output = self.noise([output, item_id])
        return output

    def summary(self):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
