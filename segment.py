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
    def __init__(self, dataPath, phrase = "deploy"):
        self.dataPath = dataPath
        self.phrase = phrase
        self.phraseSubPath = self.phrase + "/"


    #helper
    def segmentSingleFile(self, file):
        filename = os.path.basename(file)

        serializer = NoduleSerializer(self.dataPath, self.phraseSubPath)
        image = serializer.readFromNpy("nodules/", filename)

        mask = lung_segmentation.segment_HU_scan_elias(image)

        serializer.writeToNpy("mask/", filename, mask)

        print("{0}".format(filename))
        # self.progressBar.update(1)

    # interface
    def segmentAllFiles(self):
        fileList = glob(os.path.join(self.dataPath, self.phraseSubPath, "nodules/*.npy"))
        # self.progressBar = tqdm(total = len(fileList))
        pool = Pool()
        pool.map(self.segmentSingleFile, fileList)

if __name__ == "__main__":
    seg = Segment("d:/project/tianchi/data/", "deploy")
    seg.segmentAllFiles()
