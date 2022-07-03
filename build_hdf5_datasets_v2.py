from model.hdf5datasetwriter import HDF5DatasetWriter
from model.aspectawarepreprocessor import AspectAwarePreprocessor
import cv2
import imutils
import pandas as pd
import numpy as np
import progressbar

def preprocess(image):
    # resize to 128x128x3
    # to grayscale
    aap = AspectAwarePreprocessor(128, 128)
    image = aap.preprocess(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    return image

# TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
# VALID_HDF5 = "datasets/hdf5/val.hdf5"
# TEST_HDF5 = "datasets/hdf5/test.hdf5"

# TRAIN_HDF5 = "datasets/hdf5/train_position.hdf5"
# VALID_HDF5 = "datasets/hdf5/val_position.hdf5"
# TEST_HDF5 = "datasets/hdf5/test_position.hdf5"

# TRAIN_HDF5 = "dataset_google/hdf5/train_google.hdf5"
# VALID_HDF5 = "dataset_google/hdf5/val_google.hdf5"
# TEST_HDF5 = "dataset_google/hdf5/test_google.hdf5"

TRAIN_HDF5 = "dataset_varying_height/hdf5/train.hdf5"
VALID_HDF5 = "dataset_varying_height/hdf5/val.hdf5"
TEST_HDF5 = "dataset_varying_height/hdf5/test.hdf5"

train = pd.read_csv("lists/train.csv")
# train = pd.read_csv("lists/train_position.csv")
# train = pd.read_csv("lists/train_google.csv")
train_path1 = train["path1"]
train_path2 = train["path2"]
train_labels1 = train["delta_position"]
train_labels2 = train["z_position"]
train_labels3 = train["altitude"]

test = pd.read_csv("lists/test.csv")
# test = pd.read_csv("lists/test_position.csv")
# test = pd.read_csv("lists/test_google.csv")
test_path1 = test["path1"]
test_path2 = test["path2"]
test_labels1 = test["delta_position"]
test_labels2 = test["z_position"]
test_labels3 = test["altitude"]

validation = pd.read_csv("lists/val.csv")
# validation = pd.read_csv("lists/val_position.csv")
# validation = pd.read_csv("lists/val_google.csv")
validation_path1 = validation["path1"]
validation_path2 = validation["path2"]
validation_labels1 = validation["delta_position"]
validation_labels2 = validation["z_position"]
validation_labels3 = validation["altitude"]

datasets = [
    (train_path1, train_path2, train_labels1, train_labels2, train_labels3, TRAIN_HDF5),
    (validation_path1, validation_path2, validation_labels1, validation_labels2, validation_labels3, VALID_HDF5),
    (test_path1, test_path2, test_labels1, test_labels2, test_labels3, TEST_HDF5)]

for (paths1, paths2, labels1, labels2, labels3, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths1), 128, 128, 1), outputPath)

    # initialize progressbar
    widgets = ["Building dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths1), widgets=widgets).start()

    for (i, (path1, path2, label1, label2, label3)) in enumerate(zip(paths1, paths2, labels1, labels2, labels3)):

        image1 = cv2.imread(path1)
        image1 = preprocess(image1)

        image2 = cv2.imread(path2)
        image2 = preprocess(image2)    

        writer.add([image1], [image2], [(label1, label2, label3)])
        pbar.update(i)

    pbar.finish()
    writer.close()
