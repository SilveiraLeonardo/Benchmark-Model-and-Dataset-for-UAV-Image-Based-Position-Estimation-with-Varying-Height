from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

# Script correct for evaluation of the model in the validation and testing datasets

MODEL_PATH  = "checkpoints/model_checkpoint_train2_part2"
TRAIN_HDF5 = "dataset_varying_height/hdf5/train.hdf5"
VALID_HDF5 = "dataset_varying_height/hdf5/val.hdf5"
TEST_HDF5 = "dataset_varying_height/hdf5/test.hdf5"
BATCH_SIZE = 64
minHeight = 16.3
maxHeight = 111.47

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

print("[INFO] loading validation generator...")
valGen = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE, minHeight=minHeight, maxHeight=maxHeight)
gen_validation = valGen.generator(normalize=True)

total_error = 0
n_pairs = 0
for i in range(valGen.numImages // BATCH_SIZE):

	image_pair, label = next(gen_validation)
	predictions = model.predict(image_pair)

	label = np.reshape(label, (-1, 1))
	squared_error = np.square(predictions - label)
	total_error = total_error + np.sum(squared_error)

	n_pairs = n_pairs + image_pair[0].shape[0]

mean_squared_error = total_error / n_pairs
root_mean_squared_error = np.sqrt(mean_squared_error)
print("[INFO] Validation data MSE: {}".format(mean_squared_error))
print("[INFO] Validation data RMSE: {}".format(root_mean_squared_error))

valGen.close()

print("[INFO] loading test generator...")
testGen = HDF5DatasetGenerator(TEST_HDF5, BATCH_SIZE, minHeight=minHeight, maxHeight=maxHeight)
gen_test = testGen.generator(normalize=True)

total_error = 0
n_pairs = 0
for i in range(testGen.numImages // BATCH_SIZE):

	image_pair, label = next(gen_test)
	predictions = model.predict(image_pair)

	label = np.reshape(label, (-1, 1))
	squared_error = np.square(predictions - label)
	total_error = total_error + np.sum(squared_error)

	n_pairs = n_pairs + image_pair[0].shape[0]

mean_squared_error = total_error / n_pairs
root_mean_squared_error = np.sqrt(mean_squared_error)

print("[INFO] Testing data MSE: {}".format(mean_squared_error))
print("[INFO] Validation data RMSE: {}".format(root_mean_squared_error))

testGen.close()


