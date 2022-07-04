import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize=None, minHeight=None, maxHeight=None):
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]

        self.minHeight = minHeight
        self.maxHeight = maxHeight

        if batchSize is not None:
            self.batchSize = batchSize
        else:
            self.batchSize = self.numImages

    def generator(self, passes=np.inf, normalize=False):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images1 = self.db["images1"][i: i+self.batchSize]
                images2 = self.db["images2"][i: i+self.batchSize]
                labels_tuple = self.db["labels"][i: i+self.batchSize]

                images1 = images1.astype("float")/255.0
                images2 = images2.astype("float")/255.0

                # labels1 = ["delta_position"]
                # labels2 = ["z_position"]
                # labels3 = ["altitude"]
                labels = labels_tuple[:,0]
                z_position = labels_tuple[:,1]

                if normalize:
                    z_position = (z_position.astype("float") - self.minHeight)/(self.maxHeight - self.minHeight)

                yield ([images1, images2, z_position.astype("float")], labels.astype("float"))
            
            epochs += 1
    
    def close(self):
        self.db.close()
                