# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from NoduleSerializer import NoduleSerializer
import lung_segmentation

# create lung mask

class Segment(object):
    # constructor
    def __init__(self, dataPath, phase = "deploy"):
        self.dataPath = dataPath
        self.phase = phase
        self.phaseSubPath = self.phase + "/"


    #helper
    def segmentSingleFile(self, file):
        filename = os.path.basename(file)

        serializer = NoduleSerializer(self.dataPath, self.phaseSubPath)
        image = serializer.readFromNpy("resamples/", filename)

        mask = lung_segmentation.segment_HU_scan_elias(image)
        serializer.writeToNpy("mask/", filename, mask)

        image = image * mask
        serializer.writeToNpy("lung/", filename, image)

        print("{0}".format(filename))
        # self.progressBar.update(1)

    # interface
    def segmentAllFiles(self):
        fileList = glob(os.path.join(self.dataPath, self.phaseSubPath, "resamples/*.npy"))
        # self.progressBar = tqdm(total = len(fileList))
        pool = Pool()
        pool.map(self.segmentSingleFile, fileList)

if __name__ == "__main__":
    seg = Segment("d:/project/tianchi/data/", "deploy")
    seg.segmentAllFiles()
