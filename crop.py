# -*- coding:utf-8 -*-

from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

dataPath = "d:/project/tianchi/data/"
phrase = "train"

cropSize = 0
if phrase == "train":
    cropSize = 128
elif phrase == "test":
    cropSize = 64
else:
    cropSize = 64

cropper = NoduleCropper(dataPath = dataPath, phrase = phrase, cropSize = cropSize)
serializer = NoduleSerializer(dataPath = dataPath, phrase = phrase)

mhdNodules = cropper.cropAllNoduleForMhd()
for fileNodules in tqdm(mhdNodules):
    for idx, nodule in enumerate(fileNodules["nodules"]):
        serializer.writeToNpy("nodules/", fileNodules["seriesuid"] + "-" + str(idx) + ".npy", nodule)
    for idx, groundTruth in enumerate(fileNodules["groundTruths"]):
        serializer.writeToNpy("groundTruths/", fileNodules["seriesuid"] + "-" + str(idx) + ".npy", groundTruth)
