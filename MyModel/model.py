import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.experimental_run_functions_eagerly(True)
np.random.seed(0)


class LR_layer(Layer):
    def __init__(self, user_num, item_num, highest_score):
        # x_u:indicates the highest score for the historical scores of the item
        # x_i:highest rated score in user u’s historical scores
        super(LR_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.user_hs, self.item_hs = highest_score

    def build(self, input_shape):
        self.beta_u = self.add_weight(name='beta_user_vector',
                                      shape=(self.user_num, 1),
                                      initializer=tf.random_normal_initializer(seed=0),
                                      trainable=True)
        self.bias_u = self.add_weight(name='bias_user_vector',
                                      shape=(self.user_num, 1),
                                      initializer=tf.random_normal_initializer(seed=0),
                                      trainable=True)
        self.beta_i = self.add_weight(name='beta_item_vector',
                                      shape=(self.item_num, 1),
                                      initializer=tf.random_normal_initializer(seed=0),
                                      trainable=True)
        self.bias_i = self.add_weight(name='bias_item_vector',
                                      shape=(self.item_num, 1),
                                      initializer=tf.random_normal_initializer(seed=0),
                                      trainable=True)

        self.user_weight = self.add_weight(name='user_weight',
                                           shape=(self.user_num, 1),
                                           initializer=tf.random_normal_initializer(seed=0),
                                           trainable=True)
        self.item_weight = self.add_weight(name='item_weight',
                                           shape=(self.item_num, 1),
                                           initializer=tf.random_normal_initializer(seed=0),
                                           trainable=True)

    def call(self, inputs, **kwargs):
        user_id, item_id = inputs

        Xu = tf.cast(tf.nn.embedding_lookup(params=self.user_hs, ids=user_id - 1), dtype=tf.float32)
        user_beta = tf.nn.embedding_lookup(params=self.beta_u, ids=user_id - 1)
        user_bias = tf.nn.embedding_lookup(params=self.bias_u, ids=user_id - 1)
        Xi = tf.cast(tf.nn.embedding_lookup(params=self.item_hs, ids=item_id - 1), dtype=tf.float32)
        item_beta = tf.nn.embedding_lookup(params=self.beta_i, ids=item_id - 1)
        item_bias = tf.nn.embedding_lookup(params=self.bias_i, ids=item_id - 1)

        a = tf.nn.embedding_lookup(params=self.user_weight, ids=user_id - 1)
        b = tf.nn.embedding_lookup(params=self.item_weight, ids=item_id - 1)

        Yu = tf.multiply(user_beta, tf.reshape(Xu, shape=(-1, 1))) + user_bias
        Yi = tf.multiply(item_beta, tf.reshape(Xi, shape=(-1, 1))) + item_bias
        outputs = tf.multiply(a, Yu) + tf.multiply(b, Yi)
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        tf.keras.Model(inputs=[user_id, item_id], outputs=self.call([user_id, item_id])).summary()


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MyModel-old Layer
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
                                 initializer=tf.random_normal_initializer(seed=0),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(seed=0),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(seed=0),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='item_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(seed=0),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs):
        user_id, item_id = inputs

        # MyModel-old
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id - 1)  # 选取一个张量里面索引对应的元素
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id - 1)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=1, keepdims=True)  # 用户对物品的评分

        # MyModel-old-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id - 1)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id - 1)
        bias = tf.reshape((user_bias + item_bias), shape=(-1, 1))
        # use bias
        outputs = bias + outputs if self.use_bias else outputs
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        tf.keras.Model(inputs=[user_id, item_id], outputs=self.call([user_id, item_id])).summary()


class MyModel(tf.keras.Model):
    def __init__(self, feature_columns, dense_feature, use_bias=True, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MyModel-old Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param implicit: whether implicit or not
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MyModel, self).__init__()
        self.sparse_feature_columns = feature_columns
        highest_score, avg_score = dense_feature
        num_users, num_items = self.sparse_feature_columns[0]['feat_num'], \
                               self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        self.u_avg, self.i_avg = avg_score
        self.mf_layer = MF_layer(num_users, num_items, latent_dim, use_bias,
                                 user_reg, item_reg, user_bias_reg, item_bias_reg)
        self.lr_layer = LR_layer(num_users, num_items, highest_score)

    def call(self, inputs):
        sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        outputs = tf.multiply(self.mf_layer([user_id, item_id]), tf.constant(0.8)) + tf.multiply(
            self.lr_layer([user_id, item_id]), tf.constant(0.2)) \
                  + tf.reshape(
            tf.cast(tf.nn.embedding_lookup(params=self.u_avg, ids=user_id - 1) +
                    tf.nn.embedding_lookup(params=self.i_avg, ids=item_id - 1), dtype='float32'),
            shape=(-1, 1))  # 前一层的输出，前一层的预测
        return outputs

    def summary(self):
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
