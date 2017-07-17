# -*- coding:utf-8 -*-

from NoduleCropper import NoduleCropper

dataPath = "d:/project/tianchi/data/"
phase = "train"
net = "vnet"

cropper = NoduleCropper(dataPath = dataPath, phase = phase)
if (phase != "deploy") and (net == "vnet"):
    cropper.resampleAndCreateGroundTruth()
else:
    cropper.resampleAllFiles()
