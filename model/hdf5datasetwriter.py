import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied outputPath already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)

        # create two datasets: one to store the images and another to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data1 = self.db.create_dataset("images1", dims, dtype="float")
        self.data2 = self.db.create_dataset("images2", dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="float")

        self.bufSize = bufSize
        self.buffer = {"data1": [], "data2": [], "labels": []}
        self.idx = 0

    def add(self, rows1, rows2, labels):
        self.buffer["data1"].extend(rows1)
        self.buffer["data2"].extend(rows2)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs o be flushed to disk
        if len(self.buffer["data1"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data1"])
        self.data1[self.idx:i] = self.buffer["data1"]
        self.data2[self.idx:i] = self.buffer["data2"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data1": [], "data2": [], "labels": []}

    def close(self):
        # check to see if there are any other entry in the buffer
        # that need to be flushed
        if len(self.buffer["data1"]) > 0:
            self.flush()

        self.db.close()
