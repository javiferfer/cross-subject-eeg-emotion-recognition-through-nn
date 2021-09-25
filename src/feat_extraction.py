import numpy as np

from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import simps


# Welch method
def bandpower_welch(data, sf, method, band, window_sec=None, noverlap=None, relative=False):
    band = np.asarray(band)
    low, high = band

    if window_sec is not None:
        nperseg = window_sec*sf
    else:
        nperseg = (2/low) * sf
    
    if method == 'welch':
        freqs, psd_trial = welch(data, sf, window='hann', nperseg=nperseg, noverlap=noverlap)  

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd_trial[idx_band], dx=freq_res)

    return bp

# Multitaper method
def bandpower_multitaper(data, sf, method, band, relative=False):
    band = np.asarray(band)
    low, high = band
    
    if method == 'multitaper':
        psd_trial, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                                normalization='full', verbose=0)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd_trial[idx_band], dx=freq_res)
        
    return bp


# DE method
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def bandpower_de(data, sf, method, band, relative=False):
    data_band = butter_bandpass_filter(data, lowcut=band[0], highcut=band[1], fs=frequency, order=1)
    std = np.std(data_band)
    feature = 1/2*np.log(2*np.pi*np.e*std**2)
    return feature
