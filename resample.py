# -*- coding:utf-8 -*-

from NoduleCropper import NoduleCropper

dataPath = "d:/project/tianchi/data/"
phase = "deploy"

cropper = NoduleCropper(dataPath = dataPath, phase = phase)
if phase != "deploy":
    cropper.resampleAndCreateGroundTruth()
else:
    cropper.resampleAllFiles()
