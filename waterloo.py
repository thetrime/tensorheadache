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
from precise.network_runner import Listener

samples, sample_rate = librosa.load("/tmp/fixed.wav", sr=16000)

listener = Listener("qqq.pb", -1)
copy = samples[:]
for i in (4096, 4096, 4096, 3532, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096):
    chunk = copy[:i]
    print(listener.update(chunk))
    copy = copy[i:]


# This code produces 21 outputs
# bakerloo produces 75. Hmm.
