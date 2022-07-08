from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import csv

# Script correct for evaluation of the model in the validation and testing datasets

# MODEL_PATH = "checkpoints/model_checkpoint_train2_part2"
MODEL_PATH = "checkpoints/model_checkpoint_train1_part1"

# TEST_TRAJECTORY_0 = "dataset_varying_height/hdf5/path0.hdf5"
# TEST_TRAJECTORY_1 = "dataset_varying_height/hdf5/path1.hdf5"
TEST_TRAJECTORY_2 = "dataset_varying_height/hdf5/path2.hdf5"
# TEST_TRAJECTORY_3 = "dataset_varying_height/hdf5/path3.hdf5"

# test_trajectories = [("path0", TEST_TRAJECTORY_0), ("path1", TEST_TRAJECTORY_1), 
# 					("path2", TEST_TRAJECTORY_2), ("path3", TEST_TRAJECTORY_3)]
test_trajectories = [("path2", TEST_TRAJECTORY_2)]

BATCH_SIZE = 1
minHeight = 16.3
maxHeight = 111.47

# create CSV file and write header to it
header = ['predito', 'esperado', 'erro']

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

for trajectory in test_trajectories:

	path_name, HDF5_PATH = trajectory

	with open('{}_predito_esperado.csv'.format(path_name), 'w', newline='') as f:
		writer = csv.writer(f)
		# write the header
		writer.writerow(header)

	print("[INFO] loading testing generator...")
	testGen = HDF5DatasetGenerator(HDF5_PATH, BATCH_SIZE, minHeight=minHeight, maxHeight=maxHeight)
	gen_test = testGen.generator(normalize=True)

	trajectory_length = 0
	accumulated_error = 0
	total_error = 0
	n_pairs = 0
	for i in range(testGen.numImages // BATCH_SIZE):

		image_pair, label = next(gen_test)
		
		predictions = model.predict(image_pair)
		label = np.reshape(label, (-1, 1))

		error = predictions - label
		accumulated_error = accumulated_error + np.sum(error)
		trajectory_length = trajectory_length + np.sum(label)

		n_pairs = n_pairs + image_pair[0].shape[0]

		squared_error = np.square(predictions - label)
		total_error = total_error + np.sum(squared_error)

		text = [float(predictions[0]), float(label[0]), float(error[0])]
		with open('{}_predito_esperado.csv'.format(path_name), 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(text)

	mean_squared_error = total_error / n_pairs
	root_mean_squared_error = np.sqrt(mean_squared_error)

	print("[INFO] Results for {}:".format(path_name))
	print("[INFO] Accumulated error {} m in {} images of straight line flight, comprising a trajectory of {} m...".format(accumulated_error, n_pairs, trajectory_length))
	print("[INFO] Mean error of: {} /m...".format(accumulated_error/trajectory_length))
	print("[INFO] MSE: {}".format(mean_squared_error))
	print("[INFO] RMSE: {}".format(root_mean_squared_error))

	testGen.close()


