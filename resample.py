# -*- coding:utf-8 -*-

from NoduleCropper import NoduleCropper

dataPath = "d:/project/tianchi/data/"
phrase = "test"

cropper = NoduleCropper(dataPath = dataPath, phrase = phrase)
cropper.resampleAndCreateGroundTruth()
