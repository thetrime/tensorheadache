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
from keras import backend as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

model = Sequential()

# This is a very simple model. five dense layers of 192 units (based on what Apple once admitted they use in Siri) and a sigmoid output to get a binary classifier

# First a dense layer
model.add(Dense(units=192, input_dim=756))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=192))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=192))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=192))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=192))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# And now an output layer
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01), metrics=["accuracy"])

# Now we need some data. There are 3 options here:
# 1) retrieve_cached_data() returns the saved data from git
def retrieve_cached_data():
        return np.load("train-x.npy"), np.load("train-y.npy"), np.load("test-x.npy"), np.load("test-y.npy")


# 2) Extract the data from the raw samples into two arrays
def load_the_data(whence):
        inputs = []
        outputs = []
        for stub in os.listdir(whence):
                filename = os.path.join(whence, stub)
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
        # Compute the lowest 12 MFCCs
        mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=12)
        # Make into a list
        mfccs = mfccs.T.flatten()
        # Normalize the data. This seems to make an ENORMOUS difference
        return mfccs / np.linalg.norm(mfccs)


def predict(filename):
    print(model.predict(np.array([load_features(filename)])))

# And now, we can train!
# With a generator-based trainer
#model.fit_generator(data_generator(), epochs=100, steps_per_epoch=16)

# With the whole dataset
#x, y = load_the_data("training")
#xt, yt = load_the_data("testing")

x, y, xt, yt = retrieve_cached_data()

model.fit(x=x, y=y, epochs=100)

print("Testing the model....")

score = model.evaluate(x=xt, y=yt)

print("Result: ", score)

print("Saving the Keras model...")

model.save('model.h5')

print("Saved. Saving the tensorflow model...");

session = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [model.outputs[0].op.name])
graph_io.write_graph(constant_graph, ".", "model.pb", as_text=False)

print("Saved")

