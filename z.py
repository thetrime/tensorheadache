import audioread
import os
import numpy as np
import librosa
import scipy.fftpack as fft
import scipy
from librosa import util


samples, rate = librosa.load("testing/negative-00.wav", sr=16000)
s1 = librosa.core.stft(samples, n_fft=2048, hop_length=512, window='hann', center=False)
my_spectrogram = np.abs(librosa.core.stft(samples, n_fft=2048, hop_length=512))**2
their_spectrogram, _ = librosa.core.spectrum._spectrogram(y=samples, S=None, n_fft=2048, hop_length=512, power=2)

# Ok, so far we agree. My spectrogram and their output of _spectrogram are the same.
mel_basis = librosa.filters.mel(sr=16000, n_fft=2048)
my_melspectrogram = np.dot(mel_basis, my_spectrogram)
their_melspectrogram = librosa.feature.melspectrogram(samples, sr=16000)

S = librosa.core.spectrum.power_to_db(my_melspectrogram)
my_mfcc = scipy.fftpack.dct(S, axis=0, type=2, norm='ortho')[:26]
their_mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=26)
