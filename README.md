# Benchmark-Model-and-Dataset-for-UAV-Image-Based-Position-Estimation-with-Varying-Height

## Dataset Generation and Pre-Processing Files

### For generating the data:

1) path_modified_v6.py: controls a drone in the AirSim environmnent and stores the images taken from the drone's camera. This script was used to generate the images used in the training, validation and test set.
2) generate_test_paths.py: also controls a drone in the AirSim environment and stores the images taken from it. This script was used to generate test trajectories in order to evaluate the accumulated error of the model.

### For preparing the data for training, validation and testing:

1) dataset_preprocessing_2.py: creates a CSV file with all the image pairs paths and txt annotation files paths, the label of the image pair (the displacement of the drone in meters between the pair of images), and the height at which the images were taken.
2) build_data_splits_v2.py: consumes the CSV file created by the data_preprocessing_2.py, and creates 3 CSVs: train.csv, val.csv and test.csv. Each file contains the path to the pair of images and the label for each of the data splits.
3) build_hdf5_datasets_v2.py: consumes the CSV files train.csv, val.csv and test.csv, and create 3 .hdf5 databases (training, testing and validation), each containing: pair of images, resized to 128x128 and grayscaled, the height of which the images were taken, and their labels: ([image1], [image2], [height], [label]).
4) find_max_min_height.py: saves the maximum and minimum height of flight of the drone in the training split. These values are used to normalize the height value before feeding it to the neural network.

### For generating the test trajectories:

1) dataset_preprocessing_test_paths.py: creates a CSV file with all the image pairs paths, their labels (the displacement of the drone in meters between the pair of images), and the height at which the images were taken.
2) build_hdf5_test_paths.py: consumes the CSV files generated by dataset_preprocessing_test_paths.py, resizes the images to 128x128 and grayscaled, and create a .hdf5 databases containing the pre-processed images of the test trajectory as well as their labels and height information.

### Dataset Information:
* Size of training set: 117009
* Size of validation set: 14627
* Size of test set: 14626
* Minimum height on training set: 16.3 m
* Maximum height on training set: 111.47 m

## Training the network

All the python scripts for training, as well as the .hdf5 dataset files containing the preprocessed images are available at the shared drive: 

https://drive.google.com/drive/folders/0ABZGtBxMAivfUk9PVA

The directory and file structure of the drive is as follows:
```
|-> train_network_v2.py
|-> evaluate_model.py
|-> testing_trajectory_error.py
|----> checkpoints
|----> plots
|----> dataset_varying_height
         |----> hdf5
                 |-> train.hdf5
                 |-> val.hdf5
                 |-> test.hdf5
                 |-> paht2.hdf5
|----> lists
         |-> train.csv 
         |-> val.csv 
         |-> test.csv
         |-> path2.csv
|----> model
         |-> __init__.py
         |-> hdf5datasetgenerator.py
         |-> siamese_network.py
         |-> trainingMonitor.py      
```

Once the .hdf5 files containing the data are fairly large (more than 30GB), the easier way to use the code is if you have a Google Colab Pro account. If this is the case, you can mount the Google Drive in your Colab session, and execute the code using the Colab session terminal. If you don't have a Colab Pro account, it is necessary to download the data.


To train the network from scratch, use the following command:
```
python train_network_v2.py
```
Checkpoints of your model will be saved at the `checkpoints` folder, and at the end of every epoch a plot of the training and validation losses will be save at the `plots` folder.


To evaluate the best checkpoint saved, use the following command:
```
python evaluate_model.py
```
This script will currently evaluate the model checkpoint named `model_checkpoint_train2_part2`, which was the best model I found during training. To evaluate another checkpoint, it is necessary to change the global variable MODEL_PATH in the script.


Additionally, the script `testing_trajectory_error.py` is used to test the trained model on a trajectory generated in order to evaluate the accumulated error during flight. This script will generate a .csv file with the predicted displacement at every pair of images, the true displacement, and the corresponding error, as well as printing some error metrics to the console. To run this script, simply use the command:
```
python testing_trajectory_error.py
```

Aside from this, we highlight some details:
* The hfd5 datasets for the `train`, `validation` and `testing` splits, as well as the hdf5 dataset for the `test trajectory`, are on the folder `dataset_varying_height/hdf5`
* The folder `lists` contains the CSVs generated by the script `build_data_splits_v2.py`, and contains the information of the examples (data paths and labels) for each split
* The folder `model` contains classes used for training and evaluating the model: `hdf5datasetgenerator.py` implements the class responsible for generating batches of data from the hdf5 datasets; `siamese_network.py` builds the siamese network responsible for the first part of the network; `trainingMonitor.py` implements the callback responsible for saving plots of the training metrics during training.
