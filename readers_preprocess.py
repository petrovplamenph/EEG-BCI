# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:24:58 2019

@author: Plamen
"""
import numpy as np
import pandas as pd
import os
from obspy.signal.filter import bandpass

def cut_sequences(df, features, cutoff_beggining, seq_len, cut_step):
    """
    Cut the sequences
    """
    #pdb.set_trace()
    sequences = df.groupby('instance_id')[features].apply(lambda ch:ch[cutoff_beggining:])
    labels =  df.groupby('instance_id')['class'].mean() 
    
    pre_filtter_signals = []
    annotations = []
    
    indexies = labels.index
    for idx in indexies:
        transposed_sample = sequences.loc[idx,:].values
        
        if seq_len:
            for seq_start in range(0, transposed_sample.shape[0]-seq_len, cut_step):
                supsample = transposed_sample[seq_start:seq_start+seq_len]
                if seq_len == supsample.shape[0]:
                    pre_filtter_signals.append(supsample)
                    annotations.append(labels.loc[idx])
        else:
            pre_filtter_signals.append(transposed_sample)
            annotations.append(labels.loc[idx])

    pre_filtter_signals = np.array(pre_filtter_signals)
    annotations  = np.array(annotations).astype(int)
    return pre_filtter_signals, annotations


def butter_bandpass_filter(data, lowcut, highcut, freq, order):
    """
    Filter out frequencies in a signal with bandpass filter
    """
    filtered_data = bandpass(data, freqmin=lowcut, freqmax=highcut, df=freq, corners=order)
    return filtered_data


def bandpass_freq_filter(all_data, lowcut=8, highcut=45, freq=512, order=6):
    """
    Pass all signals in a dataset thrue bandpass filter
    """

    result = []
    for data in all_data:
        fillterd_signal = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            new_signal = butter_bandpass_filter(signal, lowcut, highcut, freq, order)
            fillterd_signal.append(np.array(new_signal))
        result.append(np.array(fillterd_signal).T)
    return np.array(result)

def read_filter(all_paths, train_subjects,test_subject, columns_to_read, cutoff_beggining, seq_len, cut_step):
    """
    Read the dataset 5 from BCI compoetition 3 (row signals) into sequences, with possibility to make the sequncies predetermine lenght or 
    cut them on the places where the class a person have been thinking about has changed. Get the row signals and the bandpass filterd signals
    """
    #pdb.set_trace()
    map_dict = {2:0, 3:1, 7:2}

    columns = ['Fp1', 'AF3' ,'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
               'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6',
               'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz','class']
    train_data = []
    train_anots = []
    for path in all_paths:
        files = os.listdir(path)
        
        for file in files:
            if 'train' not in file:
                continue
            ascii_grid = np.loadtxt(path+"\\"+file)
            experiment_data = pd.DataFrame(ascii_grid, columns=columns)
            experiment_data = experiment_data[columns_to_read]
            diffs =(experiment_data['class'] != experiment_data['class'].shift(1))
            experiment_data['instance_id'] = diffs.cumsum()
            experiment_data['class'] = experiment_data['class'].apply(lambda x : map_dict[x])
            experiment_data_seqs, annotations = cut_sequences(experiment_data, columns_to_read[:-1], cutoff_beggining, seq_len, cut_step)
            if test_subject in file:
                test_data = experiment_data_seqs
                test_annoations = annotations
            elif len(train_subjects) == 1:
                if train_subjects[0] in file:
                    train_data = experiment_data_seqs
                    train_anots = annotations
            else:
                train_data.append(experiment_data_seqs)    
                train_anots.append(annotations)
    if len(train_subjects) > 1:
        train_data = np.concatenate(tuple(train_data))
        train_anots = np.concatenate(tuple(train_anots))
    
    train_data_filtered  = bandpass_freq_filter(train_data)
    test_data_filtered = bandpass_freq_filter(test_data)
    
    return train_data, train_data_filtered, train_anots, test_data, test_data_filtered, test_annoations

