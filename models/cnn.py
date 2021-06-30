#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 22:50:50 2021

@author: jingtao
"""


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Conv2D, Dropout, Concatenate, Reshape
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.math import top_k
import keras.backend as K


# default hyperparameters
DIM = 10
DP = 0.2


class cnn(tf.keras.Model):
    def __init__(self, n_words, dim=DIM, dropout_rate=DP):
        super(cnn, self).__init__()

        self.dim = dim
        self.reshape = Reshape((2, -1))
        self.embedding = Embedding(
            input_dim=n_words+1, output_dim=self.dim, mask_zero=True)

        self.conv1 = Conv2D(filters=1, kernel_size=(22), input_shape=(
            None, self.dim), activation='relu', padding='same')
        self.conv2 = Conv2D(filters=1, kernel_size=(22), input_shape=(
            None, self.dim), activation='relu', padding='same')
        self.conv3 = Conv2D(filters=1, kernel_size=(22), input_shape=(
            None, self.dim), activation='relu', padding='same')

        '''
    self.conv1 = Conv1D(filters=self.dim, kernel_size=(22),input_shape= (None,self.dim),activation='relu')
    self.conv2 = Conv1D(filters=self.dim, kernel_size=(22),input_shape= (None,self.dim),activation='relu')
    self.conv3 = Conv1D(filters=self.dim, kernel_size=(22),input_shape= (None,self.dim),activation='relu')
    '''

        self.concatenate = Concatenate(axis=2)
        self.out_attn = MultiHeadAttention(num_heads=2, key_dim=self.dim)
        #self.out = Dense(1,activation='relu')
        self.fc = Dense(2, activation='softmax')
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):

        p1 = inputs[:, 0]
        p2 = inputs[:, 1]
        batch = len(p1)

        # for removing padding
        p1_mask = (p1 != 0)
        p2_mask = (p2 != 0)
        #tf.print(p1_mask)
        p1 = tf.boolean_mask(
            p1, p1_mask, axis=0, name='boolean_mask'
        )
        p2 = tf.boolean_mask(
            p2, p2_mask, axis=0, name='boolean_mask'
        )
        p1 = tf.reshape(p1, [batch, -1])
        p2 = tf.reshape(p2, [batch, -1])

        #tf.print(p1)
        #tf.print(len(p1[0]))

        # end of removing padding   FAILED
        
        
        
        

        p1 = self.embedding(p1)
        p2 = self.embedding(p2)

        # cnn
        p1 = tf.expand_dims(p1, axis=-1)
        p2 = tf.expand_dims(p2, axis=-1)

        p1 = self.conv1(p1)
        p2 = self.conv1(p2)

        p1 = self.conv2(p1)
        p2 = self.conv2(p2)

        p1 = self.conv3(p1)
        p2 = self.conv3(p2)

        p1 = tf.squeeze(p1, axis=-1)
        p2 = tf.squeeze(p2, axis=-1)

        tf.summary.image("post_conv", p1)

        p1 = tf.transpose(p1, [0, 2, 1])
        p2 = tf.transpose(p2, [0, 2, 1])

        p1 = tf.math.top_k(
            p1, k=50, sorted=False, name=None
        )[0]
        p2 = tf.math.top_k(
            p2, k=50, sorted=False, name=None
        )[0]

        p1s = tf.summary.image('./logs', [p1])

        p1 = tf.transpose(p1, [0, 2, 1])
        p2 = tf.transpose(p2, [0, 2, 1])

        x1 = self.out_attn(p1, p2)
        x2 = self.out_attn(p2, p1)

        p1 = tf.math.reduce_sum(
            x1, axis=1, keepdims=True, name=None
        )

        p2 = tf.math.reduce_sum(
            x2, axis=1, keepdims=True, name=None
        )

        #p1 = self.out(p1)
        #p2 = self.out(p2)

        # if training:
        #  x = self.dropout(x, training=training)

        x = self.concatenate([p1, p2])  # 100 x dim

        interaction = self.fc(x)

        interaction = tf.reshape(interaction, [-1, 2])

        return interaction


#model = cnn()
