from __future__ import print_function, division

# from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class Plotter(object):
    # constructor
    def __init__(self, queueSize = 30):
        self.queueSize = queueSize

    # processor
    def lossAndAccuProcessor(self):
        lossFig = plt.figure()
        plt.grid(True)

        trainLoss = np.zeros(self.lossIteration - self.lossBaseIter)
        testAccu = np.zeros(self.lossIteration - self.lossBaseIter)

        for i in range(self.lossIteration - self.lossBaseIter):
            loss, accu = self.lossQueue.get()
            trainLoss[i] = loss
            testAccu[i] = accu
            self.lossCount += 1

            if np.mod(self.lossCount, self.lossInterval) == 0:
                lossFig.clf()
                lossAx = lossFig.add_subplot(1, 1, 1)
                accuAx = lossAx.twinx()

                lossAx.plot(range(self.lossBaseIter, self.lossBaseIter + i), trainLoss[0:i], "b-", label = "Loss", linewidth = 1)
                accuAx.plot(range(self.lossBaseIter, self.lossBaseIter + i), testAccu[0:i], "g-", label = "Accu", linewidth = 1)

                lossFig.show()
                plt.pause(0.00000001)
                lossFig.savefig(self.lossNetPath + "loss-accu.png")

    def dataAndLabel2DProcessor(self):
        dataFig = plt.figure()
        dataFig.show()

        dataCount = 0

        while True:
            data, label, z = self.dataQueue.get()
            dataCount += 1

            if np.mod(dataCount, self.dataInterval) == 0:
                dataFig.clf()
                axData = dataFig.add_subplot(1, 2, 1)
                axLabel = dataFig.add_subplot(1, 2, 2)

                dataSlice = np.squeeze(data[z, :, :])
                labelSlice = np.squeeze(label[z, :, :])

                axData.imshow(dataSlice, cmap = plt.cm.gray)
                axLabel.imshow(labelSlice, cmap = plt.cm.gray)
                dataFig.show()
                plt.pause(0.00000001)

    # interface
    def initLossAndAccu(self, baseIter, iteration, netPath, spacing = 10, interval = 30):
        self.lossCount = 0

        self.lossBaseIter = baseIter
        self.lossIteration = iteration
        self.lossNetPath = netPath
        self.lossSpacing = spacing
        self.lossInterval = interval
        self.lossQueue = multiprocessing.Queue(self.queueSize)

        self.lossProc = multiprocessing.Process(target = self.lossAndAccuProcessor)
        self.lossProc.daemon = True
        self.lossProc.start()

    def plotLossAndAccu(self, loss, accu):
        self.lossQueue.put(tuple((loss, accu)))

    def initDataAndLabel2D(self, interval = 20):
        self.dataInterval = interval
        self.dataQueue = multiprocessing.Queue(self.queueSize)

        self.dataProc = multiprocessing.Process(target = self.dataAndLabel2DProcessor)
        self.dataProc.daemon = True
        self.dataProc.start()

    def plotDataAndLabel2D(self, data, label, z):
        self.dataQueue.put(tuple((data, label, z)))