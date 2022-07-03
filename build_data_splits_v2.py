from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import cv2
import csv

# df = pd.read_csv("data_pairs.csv")
# df = pd.read_csv("data_pairs_position.csv")
# df = pd.read_csv("data_pairs_google.csv")
df = pd.read_csv("data_pairs_varying_height.csv")

image_pair1_paths = df["image1_path"]
image_pair2_paths = df["image2_path"]
# velocities = df["linear_velocity"]
delta_position_meters = df["delta_position_meters"]
z_position = df["z_position"]
altitude = df["altitude"]

image_tuples = []
labels = []
for i in range(len(image_pair1_paths)):
    image_tuples.append((image_pair1_paths[i], image_pair2_paths[i]))
    labels.append((delta_position_meters[i], z_position[i], altitude[i]))

print("[INFO] constructing splits...")
split = train_test_split(image_tuples, labels, test_size = 0.20, random_state=42)
(trainPaths, testValPaths, trainLabels, testValLabels) = split

split = train_test_split(testValPaths, testValLabels, test_size = 0.50, random_state=42)
(testPaths, valPaths, testLabels, valLabels) = split

print("[INFO] size of training set: {}".format(len(trainPaths)))
print("[INFO] size of validation set: {}".format(len(valPaths)))
print("[INFO] size of test set: {}".format(len(testPaths)))

# datasets = [
#     ("train", trainPaths, trainLabels, "lists/train.csv"),
#     ("val", valPaths, valLabels, "lists/val.csv"),
#     ("test", testPaths, testLabels, "lists/test.csv")]

# datasets = [
#     ("train", trainPaths, trainLabels, "lists/train_position.csv"),
#     ("val", valPaths, valLabels, "lists/val_position.csv"),
#     ("test", testPaths, testLabels, "lists/test_position.csv")]

datasets = [
    ("train", trainPaths, trainLabels, "lists/train.csv"),
    ("val", valPaths, valLabels, "lists/val.csv"),
    ("test", testPaths, testLabels, "lists/test.csv")]

# initialize the list of Red, Green, and Blue channel averages
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}...".format(outputPath))

    # create CSV file and write header to it
    header = ['index', 'delta_position', 'z_position', 'altitude', 'path1', 'path2']

    with open(outputPath, 'w', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

    # (delta_position_meters[i], z_position[i], altitude[i])
    for (i, (path, label)) in enumerate(zip(paths,labels)):
        text = [i, label[0], label[1], label[2], path[0], path[1]]
        with open(outputPath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(text)    

#         if dType == "train":
#             for image_path in path:
#                 image = cv2.imread(image_path)
#                 (b, g, r) = cv2.mean(image)[:3]
#                 R.append(r)
#                 G.append(g)            
#                 B.append(b)

# print("[INFO] serializing means")
# D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
# with open('lists/dataset_mean.json', 'w') as f:
#     json.dump(D, f)

    