import keras
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import os
import librosa
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from sonopy import mfcc_spec, chop_array, power_spec, filterbanks, safe_log, dct

LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias
def weighted_log_loss(yt, yp):
    pos_loss = -(0 + yt) * K.log(0 + yp + K.epsilon())
    neg_loss = -(1 - yt) * K.log(1 - yp + K.epsilon())
    return LOSS_BIAS * K.mean(neg_loss) + (1. - LOSS_BIAS) * K.mean(pos_loss)


qqq = keras.models.load_model("qqq.net", custom_objects={'weighted_log_loss': weighted_log_loss})

samples, sample_rate = librosa.load("testing/negative-00.wav", duration=2.0, sr=16000)
window_samples = int(sample_rate * 0.1 + 0.5)
hop_samples = int(sample_rate * 0.05 + 0.5)

mfccs = mfcc_spec(samples, sample_rate, (window_samples, hop_samples), num_filt=20, fft_size=512, num_coeffs=13)


def my_mfcc_spec(audio, sample_rate, window_stride=(160, 80),
              fft_size=512, num_filt=20, num_coeffs=13, return_parts=False):
    """Calculates mel frequency cepstrum coefficient spectrogram"""
    powers = power_spec(audio, window_stride, fft_size)
    if powers.size == 0:
        return np.empty((0, min(num_filt, num_coeffs)))
    filters = filterbanks(sample_rate, num_filt, powers.shape[1])
    mels = safe_log(np.dot(powers, filters.T))  # Mel energies (condensed spectrogram)
    mfccs = dct(mels, norm='ortho')[:, :num_coeffs]  # machine readable spectrogram
    return powers
#    mfccs[:, 0] = safe_log(np.sum(powers, 1))  # Replace first band with log energies
#    return mfccs;


z = my_mfcc_spec(samples, sample_rate, (window_samples, hop_samples), num_filt=20, fft_size=512, num_coeffs=13)

def hertz_to_mels(f):
    return 1127. * np.log(1. + f / 700.)

def mel_to_hertz(mel):
    return 700. * (np.exp(mel / 1127.) - 1.)

def correct_grid(x):
    """Push forward duplicate points to prevent useless filters"""
    offset = 0
    for prev, i in zip([x[0] - 1] + x, x):
        offset = max(0, offset + prev + 1 - i)
        yield i + offset

def filterbanks(sample_rate, num_filt, fft_len):
    """Makes a set of triangle filters focused on {num_filter} mel-spaced frequencies"""

    # Grid contains points for left center and right points of filter triangle
    # mels -> hertz -> fft indices
    grid_mels = np.linspace(hertz_to_mels(0), hertz_to_mels(sample_rate), num_filt + 2, True)
    grid_hertz = mel_to_hertz(grid_mels)
    grid_indices = (grid_hertz * fft_len / sample_rate).astype(int)
    grid_indices = list(correct_grid(grid_indices))
    
    banks = np.zeros([num_filt, fft_len])

    for i, (left, middle, right) in enumerate(chop_array(grid_indices, 3, 1)):
        banks[i, left:middle] = np.linspace(0., 1., middle - left, False)
        banks[i, middle:right] = np.linspace(1., 0., right - middle, False)

    return banks

f = filterbanks(16000, 20, 257)


# For listening:
#   Samples arrive in the update() function. These are appended to window_audio
#   If window_audio exceeds 1,600 samples (actually, window_samples: above) then:
#       vectorize_raw is called on the whole window_audio buffer to produce new_features
#       window_audio is truncated by the length of the new_features * 800. This basically uses up all the
#       audio that was converted. The 'length' of new_features is the number of 1x13 mfcc vectors, essentially - one for each hop of input
#       new_features is appended to mfccs
#       mfccs is trimmed from the other end to ensure it remains the same size: (29, 13). In other words, we always operate on 29 chunks of data, broken into 13-dim vectors
#   If use_delta is True (which it should not be) then mfccs is passed through add_deltas()
#   Finally, we call runner.run(mfccs). This just returns model.predict(inputs[np.newaxis])[0][0]

# vectorize_raw returns one vector for 1600 inputs, two for 2400, three for 3200 etc.
# In other words, (x/800 - 1). To fill the MFCC vector entirely requires 24,000 samples or 1.5s of input.

