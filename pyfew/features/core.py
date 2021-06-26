import catch22
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import yaml
import os
import pkg_resources


"""
.. module:: features
   :platform: Unix, MacOS
   :synopsis: extracts spatial and frequency domain features

.. moduleauthor:: wearables@ox
"""


######################################################################
#
#                       Simple Features
#
######################################################################
def simple_features(xyz, feats):
    feats['xAbsMed'], feats['yAbsMed'], feats['zAbsMed'] = np.abs(np.median(xyz, axis=0))
    feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)
    feats['xIQR'], feats['yIQR'], feats['zIQR'] = stats.iqr(xyz, axis=0)
    return feats


def dist_features(v, feats):
    feats['median'] = np.median(v)
    feats['min'] = np.min(v)
    feats['max'] = np.max(v)
    feats['q25'] = np.quantile(v, .25)
    feats['q75'] = np.quantile(v, .75)
    return feats

######################################################################
#
#                       Spectral features/default
#
######################################################################
def spectral_features(v, sample_rate):
    """ Spectral entropy, 1st & 2nd dominant frequencies """

    feats = {}

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend=False,
                                 average='median')

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to focus on the relevant freqs
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend='constant',
                                 average='median')

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats['f1'] = peak_freqs[peak_ranks[0]]
        feats['f2'] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = peak_freqs[peak_ranks[0]]
    else:
        feats['f1'] = feats['f2'] = 0

    return feats


def catch22_feats(v, feats):
    # Catch22 features
    vtup = tuple(v)  # catch22 takes tuple or list
    # Successive differences
    # Shannon entropy of two successive letters in equiprobable 3-letter symbolization
    feats['SB_MotifThree_quantile_hh'] = catch22.SB_MotifThree_quantile_hh(vtup)
    # Change in correlation length after iterative differencing
    feats['FC_LocalSimple_mean1_tauresrat'] = catch22.FC_LocalSimple_mean1_tauresrat(vtup)
    # Proportion of successive differences exceeding 0.04 sigma
    feats['MD_hrv_classic_pnn40'] = catch22.MD_hrv_classic_pnn40(vtup)
    # Simple temporal statistics
    # Longest period of consecutive values above the mean
    feats['SB_BinaryStats_mean_longstretch1'] = catch22.SB_BinaryStats_mean_longstretch1(vtup)
    # Nonlinear autocorrelation
    # First minimum of the automutual information function
    feats['IN_AutoMutualInfoStats_40_gaussian_fmmi'] = catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(vtup)
    # Linear autocorrelation
    # First 1 / e crossing of autocorrelation function
    feats['CO_f1ecac'] = catch22.CO_f1ecac(vtup)
    return feats


######################################################################
#
#                       Full features (TODO)
#
######################################################################





######################################################################
#
#                           Main
#
######################################################################
def load_feature_set(feature_set):
    feature_names = ['minimal', 'default', 'full']
    if feature_set not in feature_names:
        raise Exception("Feature name [%s] not available." % feature_set)

    feature_set_root = "feature_set"
    feature_set_path = pkg_resources.resource_filename('pyfew', 'features/feature_set/'+ feature_set +'.yaml')
    #feature_set_path = os.path.join(feature_set_root, feature_set+'.yaml')
    with open(feature_set_path) as file:
        feature_set = yaml.load(file, yaml.FullLoader)
    return feature_set


def peak_features(v, sample_rate, feats):
    # Signal peaks
    u = butter_filt(v, (.6, 5), fs=sample_rate, order=8)
    # Prominence 0.25 seems to be best for 0.6-5Hz, based on the RMSE map plot
    # Also, the smallest largest prominence in Rowlands walks is ~0.3,
    # meaning some walking windows would have no steps for any higher value
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['numPeaks'] = len(peaks)
    if len(peak_props['prominences']) > 0:
        feats['peakPromin'] = np.median(peak_props['prominences'])
        feats['peakProminIQR'] = stats.iqr(peak_props['prominences'])
    else:
        feats['peakPromin'] = 0
        feats['peakProminIQR'] = 0
    return feats


def extract_features(xyz, sample_rate, feature_set='default', custom_features=None):
    """Extract hand-crafted features from an epoch of tri-axial data

    :param np.array xyz:
        Data array to extract of size N x 3 where N is sample_rate * epoch_len
    :param int sample_rate:
        As the name suggests.
    :param str feature_set:
        Specify where set of features to use. Options include minimal, full default and full
    :return:
        Extracted features for an epoch
    :rtype:
        np.array
    """
    if sample_rate <= 0:
        raise Exception("Sample rate must be non-negative")
    if len(xyz.shape) > 2 or xyz.shape[1] != 3:
        raise Exception("Input shape must of N x 3. Got %s instead." % str(xyz.shape))

    feature_set = load_feature_set(feature_set)
    feats = {}

    xyz = np.clip(xyz, -3, 3)
    if feature_set['simple_feature']:
        feats = simple_features(xyz, feats)

    v = np.linalg.norm(xyz, axis=1)
    if feature_set['dist_features']:
        feats = dist_features(v, feats)

    # Spectral features
    if feature_set['spectral_features']:
        feats.update(spectral_features(v, sample_rate))
    if feature_set['catch22_feats']:
        feats = catch22_feats(v, feats)
    if feature_set['peak_features']:
        feats = peak_features(v, sample_rate, feats)

    # include custom features
    if custom_features is not None:
        for custom_feature in custom_features:
            feats = custom_feature(xyz, feats)
    return feats


def butter_filt(x, cutoffs, fs, order=8, axis=0):
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'low'
            Wn = lowcut / nyq
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y
