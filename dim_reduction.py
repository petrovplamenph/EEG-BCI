# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:29:21 2019

@author: Plamen
"""
import numpy as np
from transformers import flatten_data
from readers_preprocess import read_filter
from time_to_freq_domain import eval_psd_not_modulated
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def LeaveOneOutKnn(X, y, n_neighbors):
    """
    Leave one out validation  with KNN classifier
    """
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    results = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        corret_prediction = (y_pred[0]== y_test[0])
        results.append(corret_prediction)
    acc = np.array(results).mean()
    return acc

def iterate_over_chanels(data, anots, n_neighbors):
    """
    Evaluate the accuracy with leave one out crossvalidation for data at each chanel separately
    """

    _, _, ch= data.shape
    chanels_acc = []
    for idx in range(ch):
        X = data[:, :,idx]
        acc = LeaveOneOutKnn(X, anots, n_neighbors)
        chanels_acc.append(acc)
    return np.array(chanels_acc)

def find_best_k(data, anots, neibhours_range):
    """
    Tune K hyperparametar of KNN
    """
    
    best_k = 0
    best_acc = 0
    for n_neighbors in neibhours_range:
        accur =  iterate_over_chanels(data, anots, n_neighbors)
        mean_acc = accur.mean()
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_k = n_neighbors
    return best_k



def rank_chanels():
    """
    Ranks the chanels according to the best accuracy that can be achevied by using only one chanel data for classification
    The train and the test data a full lenghts segments(segements corresponding to about 15 sec.)
    """
    
    all_paths = [['data_bci\\row_data\\subject1\\'], ['data_bci\\row_data\\subject2\\'],['data_bci\\row_data\\subject3\\']]

    train_subjects = ['01']
    test_subject = '02'
    freq = 512

    cutoff_beggining = 0
    columns_to_read =  ['Fp1', 'AF3' ,'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
                       'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6',
                       'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz','class']
    seq_len = 0
    cut_step = 0
    num_perseg = freq
    num_overlap = int(num_perseg/2)
    min_freq=8
    max_freq=45
    k = 3

    First_iter = True
    for path in all_paths:
        train_full_data, train_full_data_filtered, train_full_anots, test_full_data, test_sliced_full_filtered, test_full_annoations = read_filter(path, train_subjects,test_subject, columns_to_read, cutoff_beggining, seq_len, cut_step)

        psd_signals = eval_psd_not_modulated(train_full_data, num_perseg, num_overlap, freq, min_freq, max_freq) 
        chanels_acc = iterate_over_chanels(psd_signals, train_full_anots, k)
        if First_iter:
            accuracy = chanels_acc
            First_iter = False
        else:
            accuracy += chanels_acc
    accuracy = accuracy/len(all_paths)
    sorted_indexies = np.argsort(accuracy)[::-1]


    #indexis_above_treshohld = [idx for idx in sorted_indexies if accuracy[idx]> min_accary]
    return sorted_indexies
    

def evalute_subset(X_train, X_test, y_train, y_test):
    """
    Evaluate a subset of features with knn(n_neighbors=3)
    """
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def select_best_chanels():
    """
    After the chanels have been ranked via knn uses recursive feature elimination with KNN wraper to determine the best subset of chanels to use.
    """
    
    
    all_paths = [['data_bci\\row_data\\subject1\\'], ['data_bci\\row_data\\subject2\\'],['data_bci\\row_data\\subject3\\']]

    train_subjects = ['01']
    test_subject = '02'
    freq = 512

    cutoff_beggining = 0
    columns_to_read =  ['Fp1', 'AF3' ,'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
                       'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6',
                       'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz','class']
    seq_len = 0
    cut_step = 0
    num_perseg = freq
    num_overlap = int(num_perseg/2)
    min_freq=8
    max_freq=45
    
    chanels_rank = rank_chanels()
    
    result = []
    for i in range(1, len(chanels_rank)):
        intermidiate_result = []
        for path in all_paths:
            train_full_data, train_full_data_filtered, train_full_anots, test_full_data, test_full_filtered, test_full_annoations = read_filter(path, train_subjects,test_subject, columns_to_read, cutoff_beggining, seq_len, cut_step)

            train_psd_signals = eval_psd_not_modulated(train_full_data, num_perseg, num_overlap, freq, min_freq, max_freq)
            test_psd_signals = eval_psd_not_modulated(test_full_data, num_perseg, num_overlap, freq, min_freq, max_freq) 

            train_psd_signals = flatten_data(train_psd_signals[:,:,chanels_rank[:i]])
            test_psd_signals = flatten_data(test_psd_signals[:,:,chanels_rank[:i]])
            
            acc = evalute_subset(train_psd_signals, test_psd_signals, train_full_anots, test_full_annoations)
            intermidiate_result.append(acc)
            
        result.append(intermidiate_result)
    #mean_subject_acc = np.array([sum(humans_acc)/len(humans_acc) for humans_acc in result])
    #best_idx = np.argmax(mean_subject_acc)

    return result, chanels_rank

"""
def rank_features(signals, anots):
    
    max_corr = 0.95
    max_pi = 0.95 
    max_neutral_pi = 0.90
    signals_flatten = flatten_data(signals)
    
    corrs = np.corrcoef(signals_flatten.T)

    _, pi_valas = chi2(signals_flatten[anots<2], anots[anots<2])
   

    _, pi_neutural1 = chi2(signals_flatten[np.logical_or(anots==0, anots==2)], anots[np.logical_or(anots==0,anots==2)])
    _, pi_neutural2 = chi2(signals_flatten[np.logical_or(anots==1,anots==2)], anots[np.logical_or(anots==1,anots==2)])

    feature_order = np.argsort(pi_valas)
    top2 = list(feature_order[:2])
    
    selected_fets = []
    top_neutaral = []
    total_number_of_features = len(feature_order)
    for i in range(2, total_number_of_features):
        feat_idx = feature_order[i]
        add_feat = True
        
        if (pi_neutural1[feat_idx] < 0.23) and (pi_neutural2[feat_idx] < 0.23):
            for selected_idx in range(i):
                test_feat_idx = feature_order[selected_idx]
                if corrs[feat_idx, test_feat_idx]>0.95:
                    add_feat = False
                    continue
            if add_feat:
                top_neutaral.append(feat_idx)
            continue
            
            
        if pi_valas[feat_idx] > max_pi:
            continue
        if (pi_neutural1[feat_idx] > max_neutral_pi) or (pi_neutural2[feat_idx] > max_neutral_pi):
            continue
            
        sub_from_corr_cutoff = 0.3*(i/total_number_of_features)
        for selected_idx in range(i):
            test_feat_idx = feature_order[selected_idx]
            if corrs[feat_idx, test_feat_idx]>max_corr-sub_from_corr_cutoff:
                add_feat = False
                continue
        if add_feat:
            selected_fets.append(feat_idx)
    
    orderd_selected_features = top2+top_neutaral+selected_fets
    return orderd_selected_features
"""


def rank_features(signals, anots):
    """
    Ranks featutures according to their p-values(ch-square test) and 
    removers features based on thier p values and removes correlated features(pearson test)
    """
    
    max_corr = 1
    max_pi = 0.95

    
    corrs = np.corrcoef(signals.T)

    _, pi_valas = chi2(signals, anots)
   

    feature_order = np.argsort(pi_valas)
    top2 = list(feature_order[:2])
    
    total_number_of_features = len(feature_order)
    selected_fets = []
    for i in range(2, total_number_of_features):
        feat_idx = feature_order[i]
        add_feat = True
        
            
        if pi_valas[feat_idx] > max_pi:
            continue
            
        sub_from_corr_cutoff = 0.4*(i/total_number_of_features)
        for selected_idx in range(i):
            test_feat_idx = feature_order[selected_idx]
            if corrs[feat_idx, test_feat_idx]>max_corr-sub_from_corr_cutoff:
                add_feat = False
                continue
        if add_feat:
            selected_fets.append(feat_idx)
    
    orderd_selected_features = top2+selected_fets
    return orderd_selected_features


def reduce_features(train_data_lst, test_data_lst,  train_anots, data_anots):
    """
    To each feature representation in a list with diffrent feature representations applys an algoritham 
    that eliminetaes features that don't correlate with the traget values and features that correlate with others
    """
    
    train_data_reduced = []
    test_data_reduced = []
    new_anots = []
    for idx in range(len(train_data_lst)):
        
        most_important_features = rank_features(train_data_lst[idx], train_anots)
        train_data_reduced.append(train_data_lst[idx][:, most_important_features])
        test_data_reduced.append(test_data_lst[idx][:, most_important_features])
        new_anots.append(data_anots[idx]+'_reduced')
    return train_data_reduced, test_data_reduced, new_anots

    
def pca_data(train_data_lst, test_data_lst, data_anots):
    """
    Apply psa to each feature representation in a list
    """
    
    train_data_pca = []
    test_data_pca = []
    new_anots = []

    for idx in range(len(train_data_lst)):
        pca = PCA(n_components=0.985)
        X_train = pca.fit_transform(train_data_lst[idx])
        train_data_pca.append(X_train)
            
        X_test = pca.transform(test_data_lst[idx])
        test_data_pca.append(X_test)
        new_anots.append(data_anots[idx]+'_pca')
    return train_data_pca, test_data_pca, new_anots