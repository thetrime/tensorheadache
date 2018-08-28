import audioread
import os
import numpy as np
import librosa
import scipy.fftpack as fft
import scipy
from librosa import util

samples, rate = librosa.load("testing/negative-00.wav", sr=16000)
spec = librosa.core.stft(samples, n_fft=2048, hop_length=512, window='hann')
s1 = librosa.core.stft(samples, n_fft=2048, hop_length=512, window='hann', center=False)

def stftz(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))
    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix

s2 = stftz(samples, n_fft=2048, hop_length=512, window='hann', center=False)


"""
Data; 6.048683e-02i + 0.000000e+00j
Data; -3.145323e-02i + 4.041574e-01j
Data; -3.935143e-01i + -6.830263e-01j
Data; 4.084088e-01i + 7.473204e-01j



>>> s2[0][0]
(0.06048683+0j)
>>> s2[1][0]
(-0.031453233+0.4041574j)
>>> s2[2][0]
(-0.39351434-0.68302625j)
>>> s2[3][0]
(0.40840876+0.74732035j)
>>> s2[4][0]
(0.17209564-0.5555018j)
