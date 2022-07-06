from asyncore import write
import airsim
import sys
import time
from datetime import datetime
import numpy as np
import cv2

from multiprocessing import Process
from multiprocessing import Value

import os


# neighborhood
# https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo
# https://www.unrealengine.com/marketplace/en-US/product/modular-neighborhood-pack/reviews

def captureImages(writeFiles, run):
	framecounter = 1

	airsim_client_images = airsim.MultirotorClient()
	airsim_client_images.confirmConnection()

	while writeFiles.value == 1:

		if not os.path.exists("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/images_{}".format(str(run.value))):  
			os.makedirs("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/images_{}".format(str(run.value)))	

		if not os.path.exists("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/notes_{}".format(str(run.value))):  
			os.makedirs("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/notes_{}".format(str(run.value)))	
			
		# if framecounter % 1 == 0:
		response = airsim_client_images.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
	
		current_datetime = datetime.now()
		img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8) 
		img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
		# for saving, you can do :
		cv2.imwrite("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/images_{}/".format(str(run.value)) + str(framecounter) 
					+ "_" + str(current_datetime.minute) + "_" + str(current_datetime.second) + "_" + str(current_datetime.microsecond) 
					+ ".png", img_rgb)

		state = airsim_client_images.getMultirotorState()

		with open("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/test_trajectories/notes_{}/".format(str(run.value)) + str(framecounter) 
					+ "_" + str(current_datetime.minute) + "_" + str(current_datetime.second) + "_" 
					+ str(current_datetime.microsecond) + ".txt", 'w') as f:
			f.write(str(state))

		framecounter += 1

if __name__ == '__main__':  

    write_run = Value('i',0)

    write_run.value = 3

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    state = client.getMultirotorState()
    print("taking off...")
    client.takeoffAsync().join()

    time.sleep(1)

    state = client.getMultirotorState()
    if state.landed_state == airsim.LandedState.Landed:
        print("take off failed...")
        sys.exit(1)

    # AirSim uses NED coordinates so negative axis is up.
    # z of -5 is 5 meters above the original launch point



#mapa 1 - z=-100, qtnd de pontos=5000, range -350 à 350	   #map: LandscapePro - Neve     22456 - usando variacao de velocidade e clima
#mapa 2 - z=-50, qtnd de pontos=5000, range -200 à 200	   #map: DowntownWest - colégio  44616 - velocidade fixa e clima fixo
#mapa 3 - z=-60, qtnd de pontos=5000, range -300 à 300     #map: STF - PackLandscapePro-Open_World - 25289 velocidade fixa e clima fixo
# total de imagens = 92361


    client.moveToZAsync(-70,1).join()

    z_base = -100
    # z_base = -60
    start_point = airsim.Vector3r(-200,-200,z_base)

    points = []
    points.append(start_point)

    print("flying to start position...")
    result = client.moveOnPathAsync(points,
                                12, 120,
                                airsim.DrivetrainType.ForwardOnly,
                                airsim.YawMode(False, 0), 20, 1).join()

    # end_point = airsim.Vector3r(+200,+200,-50)
    # end_point = airsim.Vector3r(+200,+200,-100)
    # end_point = airsim.Vector3r(+0,+0,-100)
    end_point = airsim.Vector3r(+0,+0,-60)
    points = []
    points.append(end_point)

    # set the process in the background to save images and txts
    # set the value of the write flag
    writeFiles = Value('i',1)
    writerProcess = Process(target=captureImages, args=(writeFiles, write_run))
    writerProcess.start()

    vel = int(np.random.randint(2,10))

    print("flying on path...")
    result = client.moveOnPathAsync(points,
                                vel, 120,
                                airsim.DrivetrainType.ForwardOnly,
                                airsim.YawMode(False, 0), 20, 1).join()

    # end_point = airsim.Vector3r(+200,+200,-60)
    end_point = airsim.Vector3r(+200,+200,-100)
    points = []
    points.append(end_point)
    print("path second part...")
    result = client.moveOnPathAsync(points,
                                vel, 120,
                                airsim.DrivetrainType.ForwardOnly,
                                airsim.YawMode(False, 0), 20, 1).join()

    # terminate the process in the background
    if writerProcess is not None:
        writeFiles.value = 0
        writerProcess.join()
        
    # drone will overshoot in the last point of path, so
    # bring it back to start point before landing
    print("landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("done...")
