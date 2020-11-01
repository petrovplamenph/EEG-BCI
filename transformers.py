# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:54:31 2019

@author: EMO
"""
import numpy as np


def flatten_data(data):
    """
    Flatten the data from the diffrent chanels
    """
    result = []
    for mesurements in data:
        result.append(mesurements.flatten())
    return np.array(result)


def select_one_chanel(data, ch_idx):
    """
    Get the data only at one chanel
    """
    result = []
    for mesurements in data:
        result.append(mesurements[:,ch_idx])
    return np.array(result)
                    
                                    
def transform_to_one_chanel_data(train_data_lst, test_data_lst, data_anots):
    """
    Separate the data at each chanel as a new training set, thus makeing number of chaneles datasets
    """
    num_chanles = train_data_lst[0].shape[2]
    train_data_chanels = []
    test_data_chanels = []
    new_anots = []                      
    for data_idx in range(len(train_data_lst)):
        signals = train_data_lst[data_idx]
        test_signals = test_data_lst[data_idx]
        
        for ch_idx in range(num_chanles):
            train_one_chanel =  select_one_chanel(signals, ch_idx)
            train_data_chanels.append(train_one_chanel)
            
            test_data_one_chanel = select_one_chanel(test_signals, ch_idx)
            test_data_chanels.append(test_data_one_chanel)
            new_anots.append(data_anots[data_idx]+'_'+str(ch_idx))                     
    return train_data_chanels, test_data_chanels, new_anots