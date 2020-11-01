# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:29:52 2019

@author: Plamen
"""
import numpy as np
import math
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean

from collections import Counter
from fastdtw import fastdtw

from time_domain_features import calculate_mean_vars

def eval_peaks(signals, peak_distance):
    peak_diffs = []
    for sig in signals:
        for ch_idx in range(sig.shape[1]):
            ch_sample = sig[:,ch_idx]
            peaks, _ = find_peaks(ch_sample, distance=peak_distance)
            if peaks.shape[0]>2:
                diff = peaks[2] - peaks[0]
                peak_diffs.append(diff)
    return Counter(peak_diffs)


def fast_DTWDistance(s1, s2):
    distance, path = fastdtw(s1, s2, dist=euclidean)
    return distance


def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])


def remove_var_outliers(X, y, treshold):
    mean_vars = calculate_mean_vars(X)
    outliers_mask = (mean_vars<np.percentile(mean_vars, treshold))

    X = X[outliers_mask]
    y = y[outliers_mask]
    return X, y

