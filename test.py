# Suppose our sliding window is 2s long, and each frame is 25ms.
# Input is at 16000Hz
# This means our window is 80 frames. Each frame has 12 feature coefficients, giving us an input tensor shape of 960

# This is not really right for librosa. mfcc always uses a hop_length of 512, which means our frame size is always 0.032s
# In a 2s input @ 16Khz we have 32,000 samples, or 62.5 frames (rounded to 63 frames I guess). Each frame has 12 features, giving an input shape of (756,)

import numpy as np
import os
import librosa
from keras.models import load_model

model = load_model("model.h5")

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
        # Compute the MFCCs
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
x, y = load_the_data("training")
xt, yt = load_the_data("testing")

#x, y, xt, yt = retrieve_cached_data()

score = model.evaluate(x=xt, y=yt)

print("Result: ", score)


def zz():
        samples, sample_rate = librosa.load('testing/negative-00.wav', duration=2.0, sr=16000)
        if len(samples) < 32000:
                # We must pad the file because it is too short
                samples = np.pad(samples, (0, 32000 - len(samples) % 32000), 'constant')
        # Compute the MFCCs
        mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=12)
        # Make into a list
        mfccs = mfccs.T.flatten()
        # Normalize the data. This seems to make an ENORMOUS difference
        return mfccs
