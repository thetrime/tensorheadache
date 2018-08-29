import audioread
import os
import numpy as np
import librosa
import scipy.fftpack as fft
import scipy
from librosa import util
from librosa.core.time_frequency import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from librosa.core.time_frequency import fft_frequencies, mel_frequencies
from librosa.core.time_frequency import mel_to_hz, hz_to_mel
from librosa.core.spectrum import power_to_db, _spectrogram
window = librosa.filters.get_window('hann', 2048, fftbins=True)

samples, rate = librosa.load("testing/negative-00.wav", sr=16000)


stft_output = librosa.core.stft(samples, n_fft=2048, hop_length=512, window='hann', center=True)

my_spectrogram = np.abs(stft_output)**2
their_spectrogram, _ = librosa.core.spectrum._spectrogram(y=samples, S=None, n_fft=2048, hop_length=512, power=2)

# Ok, so far we agree. My spectrogram and their output of _spectrogram are the same.
mel_basis = librosa.filters.mel(sr=16000, n_fft=2048)
my_melspectrogram = np.dot(mel_basis, my_spectrogram)
their_melspectrogram = librosa.feature.melspectrogram(samples, sr=16000, n_mels=26)

my_S = librosa.core.spectrum.power_to_db(my_melspectrogram)
their_S = librosa.core.spectrum.power_to_db(their_melspectrogram)

my_mfcc = scipy.fftpack.dct(my_S, axis=0, type=2, norm='ortho')[:26]
their_mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=26)

