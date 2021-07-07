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
from models.transformer_cnn_no_pad import transformer_cnn_no_pad
from tensorflow import keras
from optimization import *
import numpy as np
import tensorflow as tf
from keras import backend as K
import time
import math

''' training hyperparameters'''
batch_size = 16
warmup_epochs = 50
train_epochs = 100
learning_rate = 1e-4
decay_rate = 0.5
decay_interval = 10


def warmup_scheduler(epoch, lr):
    '''
    the callback function regulating the learning rate during training.
    
    
    Parameters
    ----------
    epoch : int
        current training epoch.
    lr : float
        current learning rate.

    Returns
    -------
    float
        the learning rate for the coming epoch.

    Global Parameters Used:
    -------
    warmup_epochs
    learning_rate
    decay_rate
    decay_interval    

    '''
    new_lr = 0
    epoch += 1 # make it starts from 1
    if epoch <= warmup_epochs:
        warmup_ratio = 1.0 * epoch / warmup_epochs
        new_lr = learning_rate * warmup_ratio
    else:
        if (epoch-warmup_epochs)%decay_interval == 0:
            new_lr = lr * decay_rate
        else:
            new_lr = lr #tf.math.exp(-0.1)
    
    if (epoch%10 == 0):
        print('learning rate:', new_lr)
    
    return new_lr



loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

#train_acc_metric = keras.metrics.CategoricalCrossentropy()
#val_acc_metric = keras.metrics.CategoricalCrossentropy()

train_acc_metric = tf.keras.metrics.CategoricalAccuracy() 
'''TODO check this'''
val_acc_metric = tf.keras.metrics.CategoricalAccuracy() 

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss_batch = 0
        logits_batch = tf.zeros([1,2])
        for i in range(batch_size):
            #print(tf.expand_dims(x[i],axis=0).shape)
            
            logits = model(tf.expand_dims(x[i],axis=0), training=True)
            
            logits_batch = tf.concat([logits_batch,logits],axis=0)
            #print(logits_batch.shape)

            logits = tf.reshape(logits,[2])
            
            #print(logits.shape)
            loss_sample = loss_fn(y[i], logits)
            #print(type(loss_sample))
            loss_batch+=loss_sample
            
    logits_batch=logits_batch[1:,:]

    grads = tape.gradient(loss_batch, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #print(logits_batch.shape)
    #print('y:',y.shape)    
    train_acc_metric.update_state(y, logits_batch)
    return loss_batch

@tf.function
def test_step(x, y):
    '''TODO: add validation loss'''
    val_logits = model(tf.expand_dims(x,axis=0), training=False)
    y = tf.expand_dims(y,axis=0)

    val_acc_metric.update_state(y, val_logits)




if __name__ == "__main__":    

    ''' data '''
    loader=data_loader()
    x_train,y_train,x_test,y_test, word_dict=loader.load()
    
    #x_train = x_train[:4]
    #y_train = y_train[:4]
    ''' choose a model. can only use batch_size=1 if choose "no_pad" version'''
    # pad:
    model = cnn(len(word_dict),dim=10 )
    #model = transformer(len(word_dict),dim=10)
    #model = transformer_cnn(len(word_dict),dim=10)

    #no pad:
    #model = cnn_no_pad(len(word_dict),dim=10 )
    #model = transformer_cnn_no_pad(len(word_dict),dim=10)
    #model = transformer_no_pad(len(word_dict),dim=10)
        
    
    '''trainsformer's optimizer with warm-up steps'''
    epoch_steps = math.ceil(len(x_train)/batch_size)
    #optimizer,lr_schedule =  create_optimizer(init_lr=learning_rate,
    #                                          num_warmup_steps=warmup_epochs*epoch_steps, 
    #                                          num_train_steps=train_epochs*epoch_steps)
    
    #optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.optimizers.SGD()
    
    ''' callbacks '''
    callback = tf.keras.callbacks.LearningRateScheduler(warmup_scheduler)
    
    
    ''' make model'''
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])



    
    ''' ez train'''
    '''
    model.fit(x_train,y_train,
              validation_data=(x_test,y_test),
              batch_size=batch_size,
              callbacks=[callback],
              epochs=warmup_epochs+train_epochs)
    '''
    


    ''' custom training '''    
    ''' forward step doesn't use batch, only optimization use batch'''

    epochs = warmup_epochs+train_epochs  
    
    epoch_lr = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        
        epoch_lr = warmup_scheduler(epoch, epoch_lr)
        print('lr:',epoch_lr)
        
        K.set_value(model.optimizer.learning_rate, 0.001)
        loss = 0

        for step in range(epoch_steps-1): # 
            x = x_train[step*batch_size:(step+1)*batch_size]
            y = y_train[step*batch_size:(step+1)*batch_size]
            loss += train_step(x, y)
            '''TODO: ADD LAST BATCH HANDLING'''

        print("Total training loss:", str(loss) )

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step in range(len(x_test)):
            #x = x_test[step*batch_size:(step+1)*batch_size]
            #y = y_test[step*batch_size:(step+1)*batch_size]
            x = x_test[step]
            y = y_test[step]
            test_step(x, y)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        
    
    
    
    
    
    
    
    print('finished training')
    
    ''' evaluation '''
    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=6)
    

