#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:33:28 2021

@author: jingtao
"""

from utils import data_loader
from models.cnn import cnn
from models.cnn_no_pad import cnn_no_pad
from models.transformer import transformer
from models.transformer_cnn import transformer_cnn
from models.transformer_no_pad import transformer_no_pad
from tensorflow import keras
from optimization import *
import numpy as np
import tensorflow as tf



''' training hyperparameters'''
batch_size = 1
warmup_epochs = 100
train_epochs = 100
learning_rate = 1e-3








if __name__ == "__main__":
    




    ''' data '''
    loader=data_loader()
    x_train,y_train,x_test,y_test, word_dict=loader.load()
    
    
    ''' choose a model. can only use batch_size=1 if choose "no_pad" version'''
    # pad:
    #model = cnn(len(word_dict),dim=10 )
    #model = transformer(len(word_dict),dim=10)
    #model = transformer_cnn(len(word_dict),dim=10)
    model = transformer_no_pad(len(word_dict),dim=10)
    
    #no pad:
    #model = cnn_no_pad(len(word_dict),dim=10 )

    
    '''trainsformer's optimizer with warm-up steps'''
    epoch_steps = len(x_train)//batch_size
    optimizer,lr_schedule =  create_optimizer(init_lr=learning_rate,
                                              num_warmup_steps=warmup_epochs*epoch_steps, 
                                              num_train_steps=train_epochs*epoch_steps)
    
    ''' make model'''
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])



    
    ''' ez train'''
    model.fit(x_train,y_train,
              validation_data=(x_test,y_test),
              batch_size=batch_size,
              epochs=warmup_epochs+train_epochs)

    
    
    print('finished training')
    
    ''' evaluation '''
    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=6)
    

