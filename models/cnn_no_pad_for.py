#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:05:42 2021

@author: jingtao
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Conv2D, Dropout, Concatenate, Reshape
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.math import top_k
import keras.backend as K
import keras

# default hyperparameters
DIM = 10
DP = 0.2


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}



class cnn_no_pad(tf.keras.Model):
    def __init__(self, n_words, dim=DIM, dropout_rate=DP):
        super(cnn_no_pad, self).__init__()

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

        p1 = inputs[:, 0] # (batch,len)
        p2 = inputs[:, 1] # (batch,len)
        batch = len(p1)
        

        p1_mask = (p1 != 0)
        p2_mask = (p2 != 0)
        
        #record the length of proteins in the batch


        len1 =tf.reduce_sum(tf.cast(p1_mask,tf.int32),axis=1)
        len2 =tf.reduce_sum(tf.cast(p2_mask,tf.int32),axis=1)
        
        
        #tf.print(p1_mask)
        p1 = tf.boolean_mask(
            p1, p1_mask, axis=0, name='boolean_mask'
        ) # 1d tensor
        p2 = tf.boolean_mask(
            p2, p2_mask, axis=0, name='boolean_mask'
        )


        p1_batch = []
        p2_batch = []

        offset1 = 0
        offset2 = 0
        for i in range(batch):
            p1_batch.append(p1[offset1:offset1+len1[i]])
            p2_batch.append(p1[offset2:offset2+len2[i]])
            offset1+=len1[i]
            offset2+=len2[i]
            

        rep1 = []
        rep2 = []
        

        for i in range(int(batch)):
            tf.print('front')
            tf.print(tf.gather(p1_batch,i))
            tf.print('back')
            tf.print(len(tf.gather(p1_batch,i)))
            
            
            p1i = tf.reshape(tf.gather(p1_batch,i), [1,-1])
            p2i = tf.reshape(tf.gather(p2_batch,i), [1,-1])
            
            p1i = self.embedding(p1i)
            p2i = self.embedding(p2i)
        

        
            # cnn
            p1i = tf.expand_dims(p1i, axis=-1)
            p2i = tf.expand_dims(p2i, axis=-1)
            
            p1i = self.conv1(p1i)
            p2i = self.conv1(p2i)

            p1i = self.conv2(p1i)
            p2i = self.conv2(p2i)

            p1i = self.conv3(p1i)
            p2i = self.conv3(p2i)

            p1i = tf.squeeze(p1i, axis=-1)
            p2i = tf.squeeze(p2i, axis=-1)

            #tf.summary.image("post_conv", p1i)

            p1i = tf.transpose(p1i, [0, 2, 1])
            p2i = tf.transpose(p2i, [0, 2, 1])

            p1i = tf.math.top_k(
                p1i, k=50, sorted=False, name=None
                )[0]
            p2i = tf.math.top_k(
                p2i, k=50, sorted=False, name=None
                )[0]

            #p1s = tf.summary.image('./logs', [p1])

            p1i = tf.transpose(p1i, [0, 2, 1])
            p2i = tf.transpose(p2i, [0, 2, 1])
            
            x1 = self.out_attn(p1i, p2i)
            x2 = self.out_attn(p2i, p1i)
            
            p1i = tf.math.reduce_sum(
                x1, axis=1, keepdims=True, name=None
            )

            p2i = tf.math.reduce_sum(
                x2, axis=1, keepdims=True, name=None
            )
            
            tf.squeeze(p1i,0)
            tf.squeeze(p2i,0)
            
            rep1.append(p1i)
            rep2.append(p2i)
        
        rep1=tf.convert_to_tensor(rep1)
        rep2=tf.convert_to_tensor(rep2)
        #p1 = self.out(p1)
        #p2 = self.out(p2)

        # if training:
        #  x = self.dropout(x, training=training)

        x = self.concatenate([rep1, rep2])  # 100 x dim

        interaction = self.fc(x)

        interaction = tf.reshape(interaction, [-1, 2])

        return interaction
