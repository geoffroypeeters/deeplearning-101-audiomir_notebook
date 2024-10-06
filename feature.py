import numpy as np
import librosa


f_log = lambda data_m: np.log(1 + 10000*data_m)
flog = lambda x: np.log(1+1000*x)-np.log(1+1000)

def f_get_lms(audio_v, sr_hz, param_lms):
    """
    description:
        compute Log-Mel-Sepctrogram audio features
    inputs:
        - audio_v
        - sr_hz
    outputs:
        - data_m (nb_dim, nb_frame): Log-Mel-Spectrogram matrix
        - time_sec_v (nb_frame): corresponding time [in sec] of analysis windows
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
    description:
        compute Harmonic CQT
    inputs:
        - audio_v
        - sr_hz
    outputs:
        - data_3m (H, nb_dim, nb_frame): Harmonic CQT
        - time_sec_v (nb_frame): corresponding time [in sec] of analysis windows
        - frequency_hz_v (nb_dim): corresponding frequency [in Hz] of CQT channels
    """
    for idx, h in enumerate(param_hcqt.h_l):
        A_m = np.abs(librosa.cqt(y=audio_v, sr=sr_hz, 
                                fmin=h*param_hcqt.FMIN, 
                                hop_length=param_hcqt.HOP_LENGTH, 
                                bins_per_octave=param_hcqt.BINS_PER_OCTAVE, 
                                n_bins=param_hcqt.N_BINS))
        if idx==0: 
            data_3m = np.zeros((len(param_hcqt.h_l), A_m.shape[0], A_m.shape[1]))
        data_3m[idx,:,:] = A_m
    
    n_times = data_3m.shape[2]
    time_sec_v = librosa.frames_to_time(np.arange(n_times), 
                                            sr=sr_hz, 
                                            hop_length=param_hcqt.HOP_LENGTH)
    frequency_hz_v = librosa.cqt_frequencies(n_bins=param_hcqt.N_BINS, 
                                                    fmin=param_hcqt.FMIN, 
                                                    bins_per_octave=param_hcqt.BINS_PER_OCTAVE)

    return data_3m, time_sec_v, frequency_hz_v





def f_get_patches(T, L, S):
    """
    description
        create a list of segments/chunk/patches considering a frame-analysis of length T, each patch has lenght L and hop-size S
    """
    # --- patch_d.L_frame+(nb_patch-1)*patch_d.STEP_frame < nb_frame
    nb_patch = int(np.floor((T - L)/S + 1))
    return [{'start_frame': (num_patch*S), 'end_frame': (num_patch*S)+L} for num_patch in range(nb_patch)]