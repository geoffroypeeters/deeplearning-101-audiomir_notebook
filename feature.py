#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: feature.py
Description: A set of function for audio feature extraction
Author: geoffroy.peeters@telecom-paris.fr
"""


import numpy as np
import librosa


f_log = lambda data_m: np.log(1 + 10000*data_m)
flog = lambda x: np.log(1+1000*x)-np.log(1+1000)


def f_get_lms(audio_v, sr_hz, param_lms):
    """
    Compute Log-Mel-Sepctrogram audio features

    Parameters
    ----------
    audio_v
        np.array containing the wavform
    sr_hz
        sampling rate in Hz
    param_lms
        dictionary containing the parameters to compute the LMS
    
    Returns
    -------
    data_m 
        2D-tensor of shape (nb_dim, nb_frame) containing the Log-Mel-Spectrogram
    time_sec_v 
        np.array of shape (nb_frame) provding corresponding time [in sec] of analysis windows
    """
    # --- data (nb_dim, nb_frames)
    mel_data_m = librosa.feature.melspectrogram(y=audio_v, sr=sr_hz,
                                                n_mels=param_lms.nb_band,
                                                win_length=param_lms.L_n,
                                                hop_length=param_lms.STEP_n)
    data_m = f_log(mel_data_m)
    nb_frame = data_m.shape[1]
    time_sec_v = librosa.frames_to_time(frames=np.arange(nb_frame),
                                        sr=sr_hz,
                                        hop_length=param_lms.STEP_n)

    return data_m, time_sec_v


def f_get_hcqt(audio_v, sr_hz, param_hcqt):
    """
    Compute the Harmonic-CQT

    Parameters
    ----------
    audio_v
        np.array containing the wavform
    sr_hz
        sampling rate in Hz
    param_hcqt
        dictionary containing the parameters to compute the HCQT
    
    Returns:
    -------
    data_3m 
        3D-tensor of shape (H, nb_dim, nb_frame)containing the Harmonic-CQT
    time_sec_v (nb_frame)
        np.array of shape (nb_frame) providing corresponding time [in sec] of analysis windows
    frequency_hz_v 
        np.array of shape (nb_dim) providing the corresponding frequency [in Hz] of CQT channels
    """
    for idx, h in enumerate(param_hcqt.h_l):
        data_m = np.abs(librosa.cqt(y=audio_v, sr=sr_hz,
                                fmin=h*param_hcqt.FMIN,
                                hop_length=param_hcqt.HOP_LENGTH,
                                bins_per_octave=param_hcqt.BINS_PER_OCTAVE,
                                n_bins=param_hcqt.N_BINS))
        if idx==0:
            data_3m = np.zeros((len(param_hcqt.h_l), data_m.shape[0], data_m.shape[1]))
        data_3m[idx,:,:] = data_m

    n_times = data_3m.shape[2]
    time_sec_v = librosa.frames_to_time(np.arange(n_times),
                                            sr=sr_hz,
                                            hop_length=param_hcqt.HOP_LENGTH)
    frequency_hz_v = librosa.cqt_frequencies(n_bins=param_hcqt.N_BINS,
                                                    fmin=param_hcqt.FMIN,
                                                    bins_per_octave=param_hcqt.BINS_PER_OCTAVE)

    return data_3m, time_sec_v, frequency_hz_v


def f_get_patches(total_len, patch_len, patch_hopsize):
    """
    Create a list of segments/chunk/patches considering a frame-analysis of length T, each patch has lenght L and hop-size S

    Parameters:
    ----------
    total_len
        total number of frames
    patch_len
        length of the required patches
    patch_hopsize
        hopsize between two required patches

    Returns:
    ----------
    patch_l
        list of dictionary of type {'start_frame':, 'end_frame': }
    """
    # --- patch_len+(nb_patch-1)*patch_hopsize < total_len
    nb_patch = int(np.floor((total_len - patch_len)/patch_hopsize + 1))
    patch_l = [{'start_frame': (num_patch*patch_hopsize), 'end_frame': (num_patch*patch_hopsize)+patch_len} for num_patch in range(nb_patch)]
    return patch_l