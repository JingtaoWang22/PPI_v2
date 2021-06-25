#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:19:49 2021

@author: jingtao
"""


from collections import defaultdict
import os
import pickle
import sys
import numpy as np
from rdkit import Chem
import timeit

import copy

import matplotlib.pyplot as plt





class ppi_preprocessor:
    
    def __init__(self):
        self.word_dict = defaultdict(lambda: len(self.word_dict))
    
    def split_sequence(self, sequence, ngram): 
        ## turning sequence into words
        sequence = '-' + sequence + '='
        words = [self.word_dict[sequence[i:i+ngram]]
                 for i in range(len(sequence)-ngram+1)]
        return np.array(words)
    
    def dump_dictionary(self, dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)
    
    def ngram_dic_dataset(self, 
                          name = 'yeast',
                          dic='yeast_dic.tsv', 
                          ppi='yeast_ppi.tsv', 
                          raw_path='./data/raw/',
                          dump_path='./data/preprocessed/',
                          ngram=3):
        
        '''
            preprocess a PPI dataset with dictionary 
        '''
        
        ## reading dataset
        # protein dictionary
        dic_file=open(raw_path+dic,'r')
        dic_lines=dic_file.readlines()
        dic={}
        for i in dic_lines:
            item=i.strip().split()
            dic[item[0]]=item[1]
        dic_file.close()

        #PPIs
        ppi_file=open(raw_path+ppi,'r')
        ppi_lines=ppi_file.readlines()
        
        dataset=[]
        for i in ppi_lines:
            ppi=i.strip().split()
            p1=dic[ppi[0]]
            p2=dic[ppi[1]]
   
            w1 = self.split_sequence(p1, ngram)
            w2 = self.split_sequence(p2, ngram)
        
            #interaction=[int(ppi[2])]
            interaction = int(ppi[2])
            
            dataset.append([w1,w2,interaction])  ## int array, int array, int


        #dataset = rm_long(dataset,6000)
        np.random.shuffle(dataset)        
        
        np.save(dump_path+name,np.array(dataset))
        self.dump_dictionary(self.word_dict, dump_path + name +'_dic.pickle')
        return 





p=ppi_preprocessor()
p.ngram_dic_dataset()

