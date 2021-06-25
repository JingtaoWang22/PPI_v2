#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:34:25 2021

@author: jingtao
"""


import numpy as np
import pickle


class data_loader:
    
    def __init__(self):
        return
    
    def split_dataset(self, dataset, ratio):
        n = int(ratio * len(dataset))
        dataset_1, dataset_2 = dataset[:n], dataset[n:]
        return dataset_1, dataset_2

    def load_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    def load(self, path='./data/preprocessed/',data='yeast.npy', dic='yeast_dic.pickle', augmentation=False):
        
        
        dataset = np.load(path+data,allow_pickle=True)
        

        train, test = self.split_dataset(dataset, 0.8)
        
        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        
        for sample in train:
            
            if sample[0].shape==sample[1].shape:
                sample[0]=np.hstack([sample[0],0])
            p1=sample[0].astype(object)
            p2=sample[1].astype(object)
            pp=np.array((p1,p2)).astype(object)
            x_train.append(pp)
            y_train.append(np.array(sample[2][0]).astype(object))
            
            #if (augmentation ==True):
            #    x_train.append((sample[1].astype('float'),sample[0].astype('float')))
            #    y_train.append(float(sample[2][0]))
                
        for sample in test:
            if sample[0].shape==sample[1].shape:
                sample[0]=np.hstack([sample[0],0])
            p1=sample[0].astype(object)
            p2=sample[1].astype(object)
            pp=np.array((p1,p2)).astype(object)
            x_test.append(pp)
            y_test.append(np.array(sample[2][0]).astype(object))
        
        
        
        word_dict = self.load_pickle(path+dic)
        print(x_train[-1].dtype)
        for i in range(len(train)):
            if x_train[i].dtype!='O':
                print(x_train[i].dtype)
        print(type(x_train))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        return x_train, y_train, x_test, y_test, word_dict


'''
loader=data_loader()
x_train,y_train,x_test,y_test = loader.load()
'''