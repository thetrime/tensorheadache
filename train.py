# Suppose our sliding window is 2s long, and each frame is 25ms.
# Input is at 16000Hz
# This means our window is 80 frames. Each frame has 12 feature coefficients, giving us an input tensor shape of 960

# This is not really right for librosa. mfcc always uses a hop_length of 512, which means our frame size is always 0.032s
# In a 2s input @ 16Khz we have 32,000 samples, or 62.5 frames (rounded to 63 frames I guess). Each frame has 12 features, giving an input shape of (756,)

import numpy as np
import os
import librosa
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()

# This is a very simple model. Two dense layers of 128 units layers and a sigmoid output to get a binary classifier

# First a dense layer
model.add(Dense(units=128, input_dim=756))
model.add(Activation('relu'))
model.add(Dropout(0.8))

# And now another one, why not
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.8))

# And now an output layer
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

# Now we need some data. There are 3 options here:
# 1) retrieve_cached_data() returns the saved data from git
def retrieve_cached_data():
        return np.load("test-x.npy"), np.load("test-y.npy")


# 2) Extract the data from the raw samples into two arrays
def load_the_data():
        inputs = []
        outputs = []
        for stub in os.listdir("training"):
                filename = os.path.join("training", stub)
                inputs.append(load_features(filename))
                outputs.append(1 if stub.startswith("positive") else 0)
        return np.array(inputs), np.array(outputs)

# 3) Use a batch. This does not work very well (but is no worse than the others)
def data_generator():
        # This returns a single-item 'batch'. This is probably not good.
        while True:
                for stub in os.listdir("training"):
                        filename = os.path.join("training", stub)
                        yield (np.array([load_features(filename)]), np.array([1 if stub.startswith("positive") else 0]))

# This code extracts the very primitive features from a WAVE file
def load_features(filename):
        samples, sample_rate = librosa.load(filename, duration=2.0, sr=16000)
        if len(samples) < 32000:
                # We must pad the file because it is too short
                samples = np.pad(samples, (0, 32000 - len(samples) % 32000), 'constant')
        # Compute the MFCCs
        mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=26)
        # Keep only the lowest 12 coefficients
        mfccs = mfccs[:12]
        # Make into a list
        mfccs = mfccs.flatten()
        return mfccs


def predict(filename):
    print(model.predict(np.array([load_features(filename)])))

# And now, we can train!
# With a generator-based trainer
#model.fit_generator(data_generator(), epochs=100, steps_per_epoch=16)

# With the whole dataset
x, y = retrieve_cached_data()
model.fit(x=x, y=y, epochs=100)

