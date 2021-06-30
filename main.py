#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:33:28 2021

@author: jingtao
"""

from utils import data_loader
from models.cnn import cnn
from tensorflow import keras
from optimization import *

import tensorflow as tf







if __name__ == "__main__":
    
    loader=data_loader()
    
    x_train,y_train,x_test,y_test, word_dict=loader.load()
    
    
    
    # model



    model = cnn(len(word_dict),dim=10 )


    
    
    
    # optimization
    '''
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    '''
    
    optimizer,lr_schedule =  create_optimizer(init_lr=1e-3,num_warmup_steps=50*8000, num_train_steps=150*8000)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    

    
    
    model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=1,epochs=150)
    
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_images=True)
    #model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_test,y_test), callbacks=[tensorboard_callback])

    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=6)
    

