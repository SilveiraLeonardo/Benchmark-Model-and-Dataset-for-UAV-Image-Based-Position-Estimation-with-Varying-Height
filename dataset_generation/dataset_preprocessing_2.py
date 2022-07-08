import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from haversine import haversine, Unit

# create CSV file and write header to it
header = ['file1_name', 'file2_name', 'notes1_path', 'notes2_path', 'image1_path', 'image2_path', "delta_position_meters", "z_position", "altitude"]

with open('data_pairs_varying_height.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

dataset_dir = os.listdir("dataset_varying_height")

print("[INFO] reading files...")

# loop through the dataset directories
for dir in dataset_dir:
    # if dir == "datasets_part2.zip":
    #     continue
    files = os.listdir("dataset_varying_height/{}".format(dir))
    print("Processing: {}...".format(dir))
    # loop for the folders inside each dataset folder
    for file in files:

        if "notes" in file:
            
            note_files = os.listdir("dataset_varying_height/{}/{}".format(dir,file))
            file_names_list = []
            # loop through the files inside each subfolder of a dataset folder
            for notes in note_files:
                file_names = os.path.splitext(os.path.basename(notes))[0]
                file_names_list.append(file_names)
            
            # sort note file names
            file_names_list.sort(key=lambda x: int(x.split("_")[0]))
            
            # make file pair
            for i in range(len(file_names_list)):
                if i == (len(file_names_list) - 1):
                    continue

                file_pair1 = file_names_list[i]
                file_pair2 = file_names_list[i+1]
                
                notes_pair1_path = "dataset_varying_height/{}/{}/{}.txt".format(dir,file,file_pair1)
                notes_pair2_path = "dataset_varying_height/{}/{}/{}.txt".format(dir,file,file_pair2)            

                file_number = file.split("_")[1]
                images_pair1_path = "dataset_varying_height/{}/images_{}/{}.png".format(dir,file_number,file_pair1)
                images_pair2_path = "dataset_varying_height/{}/images_{}/{}.png".format(dir,file_number,file_pair2)

                with open(notes_pair2_path) as f:
                    v = np.zeros(shape=(2,1))
                    j = 0
                    for i,line in enumerate(f):
                        if i==15: 
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find(','))
                            latitude = float(vel[:end_of_line_index])
                            # try:
                            #     latitude = float(vel[:end_of_line_index])
                            # except:
                            #     latitude = float(vel[:(comma_index+13)])

                        if i==16: 
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find('}'))
                            longitude = float(vel[:end_of_line_index])
                            # try:
                            #     longitude = float(vel[:end_of_line_index])
                            # except:
                            #     longitude = float(vel[:(comma_index+13)])
                                                            
                    image2_lat_long = (latitude, longitude)
                    # v_mag = np.linalg.norm(v)
                with open(notes_pair1_path) as f:
                    v = np.zeros(shape=(2,1))
                    j = 0
                    for i,line in enumerate(f):
                        if i==15: 
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find(','))
                            latitude = float(vel[:end_of_line_index])
                            # try:
                            #     latitude = float(vel[:end_of_line_index])
                            # except:
                            #     latitude = float(vel[:(comma_index+13)])

                        if i==16: 
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find('}'))
                            longitude = float(vel[:end_of_line_index])
                            # try:
                            #     longitude = float(vel[:end_of_line_index])
                            # except:
                            #     longitude = float(vel[:(comma_index+13)])


                        if i==14: 
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find(','))
                            altitude = float(vel[:end_of_line_index])
                            # try:
                            #     altitude = float(vel[:end_of_line_index])
                            # except:
                            #     altitude = float(vel[:(comma_index+13)])
                                
                        if i==35:
                            _, description = line.strip().split(None, 1)
                            vel = description.split()[-1]
                            comma_index = int(vel.find('.'))
                            end_of_line_index = int(vel.find(','))
                            # print(comma_index)
                            try:
                                z_position = float(vel[:(comma_index+3)])
                            except:
                                z_position = float(vel[:(comma_index+1)])

                            # make z_position a positive number
                            z_position = z_position*(-1)

                    image1_lat_long = (latitude, longitude)
                    # v_mag = np.linalg.norm(v)
                
                delta_position = haversine(image1_lat_long, image2_lat_long, unit=Unit.METERS)
                # write to CSV data
                # ['file1_name', 'file2_name', 'notes1_path', 'notes2_path', 'image1_path', 'image2_path', "linear_velocity"]
                text = [file_pair1, file_pair2, notes_pair1_path, notes_pair2_path, images_pair1_path, images_pair2_path, delta_position, z_position, altitude]
                with open('data_pairs_varying_height.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(text)