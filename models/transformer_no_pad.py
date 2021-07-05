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
from tensorflow import keras
from tensorflow.keras import layers

# default hyperparameters
DIM = 10
DP = 0.2


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class transformer_no_pad(tf.keras.Model):
    def __init__(self, n_words, dim=DIM, dropout_rate=DP):
        super(transformer_no_pad, self).__init__()

        self.dim = dim
        self.reshape = Reshape((2, -1))
        self.embedding = Embedding(
            input_dim=n_words+1, output_dim=self.dim, mask_zero=True)

        self.encoder1 = TransformerBlock(self.dim, 2, 10)
        self.encoder2 = TransformerBlock(self.dim, 2, 10)
        self.encoder3 = TransformerBlock(self.dim, 2, 10)

        self.concatenate = Concatenate(axis=2)
        self.out_attn = MultiHeadAttention(num_heads=2, key_dim=self.dim)
        #self.out = Dense(1,activation='relu')
        self.fc = Dense(2, activation='softmax')
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):

        p1 = inputs[:, 0] # (batch,length)
        p2 = inputs[:, 1]
        batch = len(p1)
        
        
        
        p1_mask = (p1 != 0)
        p2_mask = (p2 != 0)
        
        #record the length of proteins in the batch

        
        #tf.print(p1_mask)
        p1 = tf.boolean_mask(
            p1, p1_mask, axis=0, name='boolean_mask'
        ) # 1d tensor
        p2 = tf.boolean_mask(
            p2, p2_mask, axis=0, name='boolean_mask'
        )

        p1 = tf.reshape(p1, [batch, -1])
        p2 = tf.reshape(p2, [batch, -1])
        
        
        # transformer
        p1 = self.embedding(p1)  #(batch,length,self.dim)
        p2 = self.embedding(p2)
        
        p1 = self.encoder1(p1) 
        p1 = self.encoder2(p1)
        p1 = self.encoder3(p1)
        
        p2 = self.encoder1(p2)
        p2 = self.encoder2(p2)
        p2 = self.encoder3(p2)
        


        '''concatenation'''
        p1 = tf.transpose(p1, [0, 2, 1])
        p2 = tf.transpose(p2, [0, 2, 1])
        p1 = tf.math.top_k(
            p1, k=50, sorted=False, name=None
        )[0]
        p2 = tf.math.top_k(
            p2, k=50, sorted=False, name=None
        )[0]

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
