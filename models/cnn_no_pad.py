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

    ''' customized training loop for no padding models'''
    '''

    #init
    batch_step = 1
    gradients = []
    length = len(x_train)
    step = 0
    c=0
    
    x_train = np.expand_dims(x_train,1)
    y_train = np.expand_dims(y_train,1)
    
    for step in range(length):

        x=x_train[step]
        y=y_train[step]

        
        # Open a GradientTape.
        with tf.GradientTape() as tape:
            # Forward pass.
            logits = model(x)
            # Loss value for this batch.
            loss_value = loss_fn(y, logits)

        # Get gradients of loss wrt the weights. Accumulate gradients in each batch
        gradients += tape.gradient(loss_value, model.trainable_weights)

        if ((batch_step % batch_size == 0) or step == length):
            # Update the weights of the model if at the end of the batch.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            gradients = []
            print(c,', ', end='')
            c+=1
        # increment
        batch_step += 1 
        '''
    

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

        
        #tf.print(p1_mask)
        p1 = tf.boolean_mask(
            p1, p1_mask, axis=0, name='boolean_mask'
        ) # 1d tensor
        p2 = tf.boolean_mask(
            p2, p2_mask, axis=0, name='boolean_mask'
        )

        p1 = tf.reshape(p1, [batch, -1])
        p2 = tf.reshape(p2, [batch, -1])

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

        #tf.summary.image("post_conv", p1i)

        p1 = tf.transpose(p1, [0, 2, 1])
        p2 = tf.transpose(p2, [0, 2, 1])

        p1 = tf.math.top_k(
            p1, k=50, sorted=False, name=None
            )[0]
        p2 = tf.math.top_k(
            p2, k=50, sorted=False, name=None
            )[0]

        #p1s = tf.summary.image('./logs', [p1])

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
