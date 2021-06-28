#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:33:28 2021

@author: jingtao
"""

from utils import data_loader
from models.cnn import cnn
from tensorflow import keras

import tensorflow as tf



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






if __name__ == "__main__":
    
    loader=data_loader()
    
    x_train,y_train,x_test,y_test, word_dict=loader.load()
    
    
    inputs=tf.keras.Input(shape=(2,None,))
    cnn_model = cnn(len(word_dict) )


    outputs = cnn_model(inputs)
    model = CustomModel(inputs=inputs, outputs=outputs)
    
    
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    

    
    epochs=100
    #for e in range(epochs):
    model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_test,y_test))

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=6)
    

