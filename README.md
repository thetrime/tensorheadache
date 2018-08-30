# tensorheadache
How hard can this be?

So far this is just an experiment. You can train a network by running train.py, which will read in (all!) the files in a directory called training. Files beginning with positive are considered examples of the target, and files beginning with negative are considered counterexamples.

Once trained, the model will be saved to model.pb (and model.h5)

You can then compile the C framework which will execute the model: migraine

This loads in a specific (currently hard-coded) wav file and makes a prediction based on model.pb.