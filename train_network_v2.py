# import the necessary packages
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from model.siamese_network import build_siamese_model
from model.trainingMonitor import TrainingMonitor
from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os

IMG_SHAPE = (128, 128, 1)
chanDim = -1
BATCH_SIZE = 64
EPOCHS = 30
MODEL_PATH = "checkpoints/position_pred_model"
FIG_PATH = "plots/monitor_position_{}.png".format(os.getpid())
TRAIN_HDF5 = "dataset_varying_height/hdf5/train.hdf5"
VALID_HDF5 = "dataset_varying_height/hdf5/val.hdf5"
TEST_HDF5 = "dataset_varying_height/hdf5/test.hdf5"
minHeight = 16.3
maxHeight = 111.47

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="-")
args = vars(ap.parse_args())

# configure the siamese network
print("[INFO] build network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
height = Input(shape=(1,))
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

x = concatenate([featsA, featsB], axis=chanDim)
x = Conv2D(128, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(128, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv2D(256, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(256, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv2D(512, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(512, (7,7), padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv2D(512, (8,8), padding="valid", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)

x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dense(1, activation="linear")(x)


# fuse together the output of the model and the height
x = concatenate([x, height])
output = Dense(1, activation="linear")(x)

model = Model(inputs=[imgA, imgB, height], outputs=output)

trainGen = HDF5DatasetGenerator(TRAIN_HDF5, BATCH_SIZE, minHeight=minHeight, maxHeight=maxHeight)
valGen = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE, minHeight=minHeight, maxHeight=maxHeight)

# best model checkpoint
ckp_path = "checkpoints/model_checkpoint"
mcp = ModelCheckpoint(filepath=ckp_path,
					save_weights_only=False,
					monitor="val_loss",
					save_best_only=True,
					mode="auto",
					save_freq="epoch",
					verbose=1)

callbacks=[mcp, TrainingMonitor(FIG_PATH)]

if args["model"] == "-":
	print("[INFO] compiling model...")
	mse = MeanSquaredError()
	model.compile(loss= mse , optimizer="adam", metrics=["mean_squared_error"])
else:
	print("[INFO] loading model from disk...")
	model = load_model("checkpoints/{}".format(args["model"]))

	# update the learning rate
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 0.0005)
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

print("[INFO] training model...")
gen_training = trainGen.generator(normalize=True)
gen_validation = valGen.generator(normalize=True)

model.fit(
	gen_training,
	steps_per_epoch = trainGen.numImages // BATCH_SIZE,
	validation_data= gen_validation,
	validation_steps = valGen.numImages // BATCH_SIZE,
	epochs = EPOCHS,
	max_queue_size = 10,
	callbacks = callbacks,
	verbose = 1)

print("[INFO] serializing model")
model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
