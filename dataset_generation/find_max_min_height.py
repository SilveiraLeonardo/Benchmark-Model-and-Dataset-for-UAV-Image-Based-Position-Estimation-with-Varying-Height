from model.hdf5datasetgenerator import HDF5DatasetGenerator
import numpy as np

# Script for finding the maximum and minimum height in the training data
# with the goal of normalizing the input height to the model between 0 and 1

TRAIN_HDF5 = "dataset_varying_height/hdf5/train.hdf5"
BATCH_SIZE = 64


print("[INFO] loading validation generator...")
trainGen = HDF5DatasetGenerator(TRAIN_HDF5, BATCH_SIZE)

gen_train = trainGen.generator()

min_height = 9999.9
max_height = 0.0

for i in range(trainGen.numImages // BATCH_SIZE):

	inputs, _ = next(gen_train)

	_, _, z_position = inputs

	z_max = np.max(z_position)
	z_min = np.min(z_position)

	if z_min < min_height:
		min_height = z_min

	if z_max > max_height:
		max_height = z_max

print("[INFO] Minimum height on training set: {}".format(min_height))
print("[INFO] Maximum height on training set: {}".format(max_height))

trainGen.close()