# -*- coding:utf-8 -*-

from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

dataPath = "../tianchi/data/test1/"

cropper = NoduleCropper(dataPath)
serializer = NoduleSerializer(dataPath)

mhdNodules = cropper.cropAllNoduleForMhd()
for fileNodules in tqdm(mhdNodules):
    for idx, nodule in enumerate(fileNodules["nodules"]):
        serializer.writeToMhd("nodules/", fileNodules["seriesuid"], idx, nodule)
    for idx, groundTruth in enumerate(fileNodules["groundTruths"]):
        serializer.writeToMhd("groundTruths/", fileNodules["seriesuid"], idx, groundTruth)
