
# EMD is performed to remove any segments consists noise.
from PyEMD import EMD
import biosppy
import scipy
import neurokit2 as nk
import nolds
import pywt
import numpy as np
import pandas as pd

def run_emd(signal, nb_level_decomp=5, nb_comp_retain=3):
    """
    Run Emperical Mode Decomposition for a signal.
    :param segment:
    :param nb_level_decomp:
    :param nb_comp_retain:
    :return:
    """
    emd = EMD.EMD()
    emd.FIXE = nb_level_decomp

    #try:
    imf = emd.emd(signal)
    # Based on following paper-- adding first three imfs
    # http://kresttechnology.com/krest-academic-projects/krest-major-projects/ECE/B-Tech%20Papers/67.pdf
    return np.sum(imf[:nb_comp_retain], axis=0)  # Reconstruct signal
    #except:
    #    return None


# Loop for extracting time and frequency domain HRV using rpeaks information

def extract_time_and_freq_features(signal, sampling_rate=512, return_peaks=False, run_emd_=True):

    if run_emd_:
        emd_cleaned_sig = run_emd(signal)
    else:
        emd_cleaned_sig = signal

    # Extracting rpeaks using hamilton algorithm
    peaks = biosppy.signals.ecg.ecg(signal=emd_cleaned_sig, sampling_rate=sampling_rate, show=False)[2]
    if len(peaks) == 0:
        return None

    # Extracting HRV values on the basis of rpeaks
    features = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)  # All domains time and freq.
    if return_peaks:
        return peaks, features
    return features

def run_dwt(signal, threshold = 0.15):

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(signal, 'db4', level=6)
    coeffs[1:] = [pywt.threshold(level_coeffs, threshold*np.max(level_coeffs)) for level_coeffs in coeffs[1:]]

    return pywt.waverec(coeffs, 'db4')


def extract_dwt_features(signal, threshold = 0.15, run_dwt_=True):
    """
    Calculate Detrended Fluctuation Analysis (DFA) and Sample Entropy after a filtering using the discrete
    wavelet transform.
    :param threshold: # Threshold for filtering, based on "https://arxiv.org/ftp/arxiv/papers/1703/1703.00075.pdf"
    :return:
    """
    if run_dwt_:
        signal = run_dwt(signal, threshold)

    entropy = nolds.sampen(signal)
    dfa = nolds.dfa(signal)
    return pd.DataFrame({"samp_entropy":[entropy], "dfa":[dfa]})


def extract_all_features(signal, sampling_rate=512, run_dwt_=True, run_emd_=True):
    features_hrv = extract_time_and_freq_features(signal, sampling_rate=sampling_rate, run_emd_=run_emd_)
    selected_features = ['HRV_MeanNN', 'HRV_CVNN', 'HRV_MedianNN', 'HRV_pNN20',
                         'HRV_HTI', 'HRV_SD1SD2', 'HRV_CSI', 'HRV_CVI']

    features_dwt = extract_dwt_features(signal, run_dwt_=run_dwt_)

    return pd.concat([features_hrv[selected_features], features_dwt], axis=1)
