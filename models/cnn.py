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
    self.embedding = Embedding( input_dim = n_words, output_dim = 10)
    
    self.conv1 = Conv1D(filters=10,kernel_size=(3),input_shape= (None,10))
    self.concatenate = Concatenate(axis=1)
    self.fc = Dense(2,activation='relu')
    self.dropout = Dropout(0.2)

  def call(self, inputs, training=False):
    
    #inputs = tf.squeeze(inputs,[0])

    p1 = self.embedding(inputs[:,0])
    p2 = self.embedding(inputs[:,1])
    

    
    p1 = self.conv1(p1)
    p2 = self.conv1(p2)
    
    

    
    p1=top_k(tf.transpose(p1,perm=[0,2,1]),50)[0] 
    p2=top_k(tf.transpose(p2,perm=[0,2,1]),50)[0]

    print(p1)
    print(p2)
    
    
    p1=tf.transpose(p1,perm=[0,2,1])
    p2=tf.transpose(p2,perm=[0,2,1]) # 50 x dim

    
    #if training:
    #  x = self.dropout(x, training=training)
    
    x = self.concatenate([p1,p2]) # 100 x dim
    
    print(x)
    
    return self.fc(x)



#model = cnn()











