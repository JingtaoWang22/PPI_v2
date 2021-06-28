#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 22:50:50 2021

@author: jingtao
"""



import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Conv2D, Dropout, Concatenate, Reshape
from tensorflow.math import top_k
import keras.backend as K
class cnn(tf.keras.Model):

  def __init__(self, n_words):
    super(cnn, self).__init__()
    
    self.reshape = Reshape((2,-1))
    self.embedding = Embedding( input_dim = n_words+1, output_dim = 10, mask_zero=True)
    
    self.conv1 = Conv1D(filters=10,kernel_size=(3),input_shape= (None,10))
    self.concatenate = Concatenate(axis=2)
    #self.out = Dense(1,activation='relu')
    self.fc = Dense(2,activation='softmax')
    self.dropout = Dropout(0.2)

  def call(self, inputs, training=False):
    

    p1 = self.embedding(inputs[:,0])
    p2 = self.embedding(inputs[:,1])
    

    
    p1 = self.conv1(p1)
    p2 = self.conv1(p2)
    
    
    p1 = tf.math.reduce_sum(
    p1, axis = 1, keepdims=True, name=None
    )
    
    p2 = tf.math.reduce_sum(
    p2, axis = 1, keepdims=True, name=None
    )
    

    
    #p1 = self.out(p1)
    #p2 = self.out(p2)
    
    
    #if training:
    #  x = self.dropout(x, training=training)
    
    x = self.concatenate([p1,p2]) # 100 x dim

    
    interaction = self.fc(x)
    interaction = tf.reshape(interaction,[-1,2])
    
    
    return interaction


#model = cnn()











