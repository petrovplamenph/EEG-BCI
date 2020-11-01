# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:24:19 2019

@author: Plamen
"""
import numpy as np
from scipy.integrate import simps

def eval_band_power(all_data, freq_res):
    """
    Evaluate the energy in a given band
    """
    result = []

    for data in all_data:
        band_power_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            delta_power = simps(signal, dx=freq_res)

            band_power_signals.append(delta_power)
        result.append(np.array(band_power_signals))
    return np.array(result)
