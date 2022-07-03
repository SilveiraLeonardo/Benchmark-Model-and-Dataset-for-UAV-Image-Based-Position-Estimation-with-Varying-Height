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

		if not os.path.exists("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/images_{}".format(str(run.value))):  
			os.makedirs("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/images_{}".format(str(run.value)))	

		if not os.path.exists("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/notes_{}".format(str(run.value))):  
			os.makedirs("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/notes_{}".format(str(run.value)))	
			
		# if framecounter % 1 == 0:
		response = airsim_client_images.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
	
		current_datetime = datetime.now()
		img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8) 
		img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
		# for saving, you can do :
		cv2.imwrite("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/images_{}/".format(str(run.value)) + str(framecounter) 
					+ "_" + str(current_datetime.minute) + "_" + str(current_datetime.second) + "_" + str(current_datetime.microsecond) 
					+ ".png", img_rgb)

		state = airsim_client_images.getMultirotorState()

		with open("C:/Users/drones/Documents/Leonardo_Mateus/scripts_leo_mateus/dataset_varying_height/notes_{}/".format(str(run.value)) + str(framecounter) 
					+ "_" + str(current_datetime.minute) + "_" + str(current_datetime.second) + "_" 
					+ str(current_datetime.microsecond) + ".txt", 'w') as f:
			f.write(str(state))

		framecounter += 1

if __name__ == '__main__':  

	write_run = Value('i',0)

	for i in range(50):
		write_run.value = i

		client = airsim.MultirotorClient()
		client.confirmConnection()
		client.enableApiControl(True)
		client.armDisarm(True)

		state = client.getMultirotorState()
		print("taking off...")
		client.takeoffAsync().join()
		#if state.landed_state == airsim.LandedState.Landed:
			#print("taking off...")
			#client.takeoffAsync().join()
		#else:
		#	client.hoverAsync().join()

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

		z_base = -100
		z_variation = 50
		# z = -60
		disturbance = int(np.random.randint(-10,+10))
		z_base = z_base + disturbance
		client.moveToZAsync(z_base,1).join()


		list_of_points = []
		for _ in range(5000):
			x = int(np.random.randint(-200,200))
			y = int(np.random.randint(-200,200))
			# z = int(np.random.randint(z_base - z_variation, z_base + z_variation))
			z = int(np.random.randint(z_base, z_base + z_variation))

			point = airsim.Vector3r(x,y,z)
			list_of_points.append(point)

		# set the process in the background to save images and txts
		# set the value of the write flag
		writeFiles = Value('i',1)
		writerProcess = Process(target=captureImages, args=(writeFiles, write_run))
		writerProcess.start()

		vel = int(np.random.randint(2,10))

		print("flying on path...")
		result = client.moveOnPathAsync(list_of_points,
									vel, 120,
									airsim.DrivetrainType.ForwardOnly,
									airsim.YawMode(False, 0), 20, 1).join()

		# result = client.moveOnPathAsync([airsim.Vector3r(125,0,z),
		# 								airsim.Vector3r(125,-130,z),
		# 								airsim.Vector3r(0,-130,z),
		# 								airsim.Vector3r(0,0,z)],
		# 							12, 120,
		# 							airsim.DrivetrainType.ForwardOnly,
		# 							airsim.YawMode(False, 0), 20, 1).join()

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
