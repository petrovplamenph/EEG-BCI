# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:05:12 2019

@author: Plamen
"""
import numpy as np
from scipy import fftpack
from scipy.signal import welch, spectrogram
from scipy.integrate import simps
import pdb

#FFT
def fft_signal(all_data, freq=512, lowcut=8, highcut=45):
    """
    From singnal in the time domain(A(t)) transgrom the data in to the frequncy domain E(f) with Forier transform and and square of the abc of 
    the amplitude of the frequencies, passes frequncies only between lowcut and highcut
    """
    result = []

    for data in all_data:
        fft_signal = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            num_points = signal.shape[0]
            amp_values = np.abs(fftpack.fft(signal))**2
            freq_values = fftpack.fftfreq(num_points, d=1/freq)

            freq_mask= np.logical_and(freq_values>=lowcut, freq_values<=highcut)
            #postive_sequnce_end = int((num_points+1)/2)
            new_signal = amp_values[freq_mask]
            fft_signal.append(new_signal)
        result.append(np.array(fft_signal).T)
    return np.array(result)

#Specturm
def get_psd_values(signal, num_perseg, num_overlap, freq, min_freq, max_freq):
    """
    Gets the power spectorum in a signal via the welch methood with hanning window 
    Passes frequencies only between min_freq and max_freq
    """
    freq_values, psd_values = welch(signal, fs=freq, nperseg=num_perseg, noverlap=num_overlap)
    freq_mask= np.logical_and(freq_values>=min_freq, freq_values<=max_freq)
    psd_values = psd_values[freq_mask]
    return psd_values



def eval_psd(all_data, num_perseg, num_overlap, freq, min_freq, max_freq):
    """
    Get the powere spectrum of all signals via welch with hanning window 
    """
    result = []

    for data in all_data:
        psd_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            sig_psd = get_psd_values(signal, num_perseg, num_overlap, freq, min_freq, max_freq)

            psd_signals.append(sig_psd)
        result.append(np.array(psd_signals).T)
    return np.array(result)



def get_psd_values_not_modulated(signal, num_perseg, num_overlap, freq, min_freq, max_freq):
    """
    Gets the power spectorum in a signal via the welch methood with boxcar window(it is a stepwize function) 
    Passes frequencies only between min_freq and max_freq
    """
    freq_values, psd_values = welch(signal, window='boxcar',fs=freq, nperseg=num_perseg, noverlap=num_overlap)
    freq_mask= np.logical_and(freq_values>=min_freq, freq_values<=max_freq)
    psd_values = psd_values[freq_mask]
    return psd_values



def eval_psd_not_modulated(all_data, num_perseg, num_overlap, freq, min_freq, max_freq):
    """
    Get the powere spectrum of all signals via welch with boxcar window 
    """
    result = []

    for data in all_data:
        psd_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            sig_psd = get_psd_values_not_modulated(signal, num_perseg, num_overlap, freq, min_freq, max_freq)

            psd_signals.append(sig_psd)
        result.append(np.array(psd_signals).T)
    return np.array(result)


def eval_band_power(all_data, freq_res, cut_point1, cut_point2):
    """
    Evalutes the enrgy in freq. bandas with index [0 to cut_point1], and [cut_point1 to cut_point2] and [cut_point2 to -1]
    """
    result = []

    for data in all_data:
        band_power_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            band1_power = simps(signal[:cut_point1], dx=freq_res)
            band2_power = simps(signal[cut_point1:cut_point2], dx=freq_res)
            band3_power = simps(signal[cut_point2:], dx=freq_res)
            band_power_signals.append(np.array([band1_power, band2_power, band3_power]))
        result.append(np.array(band_power_signals).T)
    return np.array(result)

################################Functions not needed in the current last version of the project, but they may be needed in future versions################################################################################################################################################################
def get_spectrum(signal, window_size, freq, overlap, min_freq, max_freq):
    freq_values, times, spectrum_values = spectrogram(signal, nperseg=window_size ,fs=freq, noverlap=overlap,window='boxcar',  scaling='spectrum')
    
    low_cut = (freq_values>min_freq)
    high_cut = (freq_values<max_freq)
    bandwith = np.logical_and(low_cut, high_cut)
    spectrum_values = spectrum_values[bandwith]
    return spectrum_values


def eval_spectrum(all_data, window_size, freq, overlap, min_freq=8, max_freq=45):
    result = []

    for data in all_data:
        spectrum_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            sig_spectrum = get_spectrum(signal, window_size, freq,overlap, min_freq, max_freq)

            spectrum_signals.append(sig_spectrum)
        result.append(np.array(spectrum_signals).T)
    return np.array(result)


def eval_short_psd(all_data, num_perseg, num_overlap, freq, min_freq, max_freq):
    result = []

    for data in all_data:
        psd_signals = []
        for signal_idx in range(data.shape[1]):
            signal = data[:,signal_idx]
            scaned_signal = []
            for window_idx in range(0, signal.shape[0]-num_perseg, num_perseg):
                segment = signal[window_idx: window_idx+num_perseg]
                segment_psd = get_psd_values(segment, num_perseg, num_overlap, freq, min_freq, max_freq)
                scaned_signal.append(segment_psd)
            psd_signals.append(np.array(scaned_signal))
        result.append(np.array(psd_signals).T)
    return np.array(result)
