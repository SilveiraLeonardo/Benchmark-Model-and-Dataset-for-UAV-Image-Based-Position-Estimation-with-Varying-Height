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

TEST_PATH0_HDF5 = "dataset_varying_height/hdf5/path0.hdf5"
TEST_PATH1_HDF5 = "dataset_varying_height/hdf5/path1.hdf5"
TEST_PATH2_HDF5 = "dataset_varying_height/hdf5/path2.hdf5"
TEST_PATH3_HDF5 = "dataset_varying_height/hdf5/path3.hdf5"

data_pairs_path0 = "lists/data_pairs_path_0.csv"
data_pairs_path1 = "lists/data_pairs_path_1.csv"
data_pairs_path2 = "lists/data_pairs_path_2.csv"
data_pairs_path3 = "lists/data_pairs_path_3.csv"

trajectories = [(data_pairs_path0, TEST_PATH0_HDF5),
        (data_pairs_path1, TEST_PATH1_HDF5),
        (data_pairs_path2, TEST_PATH2_HDF5),
        (data_pairs_path3, TEST_PATH3_HDF5)]

for trajectory in trajectories:

    data_pairs, hdf5_path = trajectory

    test = pd.read_csv(data_pairs)
    test_path1 = test["image1_path"]
    test_path2 = test["image2_path"]
    test_labels1 = test["delta_position_meters"]
    test_labels2 = test["z_position"]
    test_labels3 = test["altitude"]

    datasets = [(test_path1, test_path2, test_labels1, test_labels2, test_labels3, hdf5_path)]

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
