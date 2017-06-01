# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
from tqdm import tqdm

from NoduleSerializer import NoduleSerializer
import lung_segmentation

# create lung mask

class Segment(object):
    # constructor
    def __init__(self, dataPath, phrase = "deploy"):
        self.dataPath = dataPath
        self.phrase = phrase
        self.phraseSubPath = self.phrase + "/"
        self.serializer = NoduleSerializer(self.dataPath, self.phraseSubPath)

    # interface
    def segmentAllFiles(self):
        fileList = glob(os.path.join(self.dataPath, self.phraseSubPath, "nodules/*.npy"))
        for file in enumerate(tqdm(fileList)):
            print(file)
            filename = os.path.basename(file[1])
            image = self.serializer.readFromNpy("nodules/", filename)
            mask = lung_segmentation.segment_HU_scan_elias(image)
            self.serializer.writeToNpy("mask/", filename, image)

if __name__ == "__main__":
    seg = Segment("d:/project/tianchi/data/", "test")
    seg.segmentAllFiles()
