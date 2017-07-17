from __future__ import print_function, division

# from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from NoduleSerializer import NoduleSerializer

class Plotter(object):
    # constructor
    def __init__(self, queueSize = 30):
        self.queueSize = queueSize

    # processor
    def lossAndAccuProcessor(self):
        lossFig = plt.figure()
        lossFig.show()

        trainLoss = np.zeros(self.lossIteration - self.lossBaseIter)
        testAccu = np.zeros(self.lossIteration - self.lossBaseIter)

        for i in range(self.lossInterval - self.lossBaseIter):
            loss, accu = self.lossQueue.get()
            trainLoss[i] = loss
            testAccu[i] = accu
            self.lossCount += 1

            if np.mod(self.lossCount, self.lossSpacing) == 0:
                lossFig.clf()
                lossAx = lossFig.add_subplot(1, 1, 1)
                accuAx = lossAx.twinx()

                lossAx.plot(range(self.lossBaseIter, self.lossBaseIter + i), trainLoss[0:i], "b-", label = "Loss", linewidth = 1)
                accuAx.plot(range(self.lossBaseIter, self.lossBaseIter + i), testAccu[0:i], "g-", label = "Accu", linewidth = 1)

                lossFig.draw()
                lossFig.savefig(self.lossNetPath + "loss-accu.png")


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
        self.lossQueue.put(tuple(loss, accu))