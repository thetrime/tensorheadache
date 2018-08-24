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

input_shape = (5511, 101)

## Convolutional layer with window size of 15
#model.add(Conv1D(196, kernel_size=15, strides=4, input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.8))

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

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

## First GRU layer
#model.add(GRU(units=128, return_sequences=True))
#model.add(Dropout(0.8))
#model.add(BatchNormalization())

## Second GRU layer
#model.add(GRU(units=128, return_sequences=True))
#model.add(Dropout(0.8))
#model.add(BatchNormalization())
#model.add(Dropout(0.8))

# Finally, a time-distributed dense layer
#model.add(TimeDistributed(Dense(1, activation="sigmoid")))

#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01), metrics=["accuracy"])

# Now we need some data. For this we will use a Python generator

def load_features(path, filename):
        samples, sample_rate = librosa.load(os.path.join(path, filename), duration=2.0, sr=16000)
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


def data_generator():
        # This returns a single-item 'batch'. This is probably not good.
        while True:
                for filename in os.listdir("positive"):
                        yield (np.array([load_features("positive", filename)]), np.array([1]))
                for filename in os.listdir("negative"):
                        yield (np.array([load_features("negative", filename)]), np.array([0]))


def predict(filename):
    print(model.predict(np.array([load_features(".", filename)])))

# And now, we can train!
model.fit_generator(data_generator(), epochs=100, steps_per_epoch=16)

