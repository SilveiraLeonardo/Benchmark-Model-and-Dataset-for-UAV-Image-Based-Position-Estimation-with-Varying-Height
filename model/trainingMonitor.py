from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    
    def on_epoch_end(self, epoch, logs={}):
        for (k,v) in logs.items():
            l = self.H.get(k,[])
            l.append(float(v))
            self.H[k] = l

        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H["loss"])>1:
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, len(self.H["loss"])), self.H["loss"], label="train_loss")
            plt.plot(np.arange(0, len(self.H["loss"])), self.H["val_loss"], label="val_loss")
            # plt.plot(np.arange(0, len(H.history["loss"])), H.history["accuracy"], label="train_acc")
            # plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_accuracy"], label="val_accuracy")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            plt.savefig(self.figPath)
            plt.close()