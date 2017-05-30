# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
from tqdm import tqdm

import NoduleSerializer
import lung_segmentation

# create lung mask

class Segment(object):
    # constructor
    def __init__(self, dataPath, phrase = "deploy"):
        self.dataPath = dataPath
        self.phrase = phrase
        self.phraseSubPath = self.phrase + "/"
        self.serializer = NoduleSerializer(self.dataPath)

    # helper
    def segmentSingleFile(self, filename):

    # interface
    def segmentAllFiles(self):
        fileList = glob(os.path.join(self.dataPath, self.phraseSubPath, "resample/*.npy"))
        for file in enumerate(tqdm(fileList)):
            filename = os.path.basename(file)
            image = self.serializer.readFromNpy(self.phraseSubPath + "resample/", filename)
            mask = lung_segmentation.segment_HU_scan_elias(image)
            self.serializer.writeToNpy(self.phraseSubPath + "mask/", filename)
