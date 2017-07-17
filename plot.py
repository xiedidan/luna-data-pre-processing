from __future__ import print_function, division

# from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from NoduleSerializer import NoduleSerializer

def plot_3d(image, threshold=0):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plotMark2D(image, center, diameter):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    image = np.rint(image).astype(np.int16)
    center = np.rint(center).astype(np.int)

    z = center[0]
    imageSlice = np.squeeze(image[z, :, :])
    ax1.imshow(imageSlice, cmap=plt.cm.gray)

    groundTruthSlice = np.zeros(imageSlice.shape)
    groundTruthSlice = markNodule(groundTruthSlice, center, diameter)
    ax2.imshow(groundTruthSlice, cmap=plt.cm.gray)

    plt.pause(100)

def plotTrainSampleFromFile2D(serializer, seriesuid, number):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    image = serializer.readFromNpy("nodules/", "{0}-{1}.npy".format(seriesuid, number))
    groundTruth = serializer.readFromNpy("groundTruths/", "{0}-{1}.npy".format(seriesuid, number))

    z = np.int(np.rint(image.shape[0] / 2))

    imageSlice = np.squeeze(image[z, :, :])
    groundTruthSlice = np.squeeze(groundTruth[z, :, :])

    ax1.imshow(imageSlice, cmap=plt.cm.gray)
    ax2.imshow(groundTruthSlice, cmap=plt.cm.gray)

    plt.show()

def plotVnetCrop2D(crop, z):
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    image = crop["image"]
    groundTruth = crop["groundTruth"]

    imageSlice = np.squeeze(image[z, :, :])
    groundTruthSlice = np.squeeze(groundTruth[z, :, :])

    ax1.imshow(imageSlice, cmap=plt.cm.gray)
    ax2.imshow(groundTruthSlice, cmap=plt.cm.gray)
    plt.show()

# helper
def markNodule(imageSlice, center, diameter, crossSize=2):
    # draw cross
    # xRange = [center[2] - crossSize, center[2] + crossSize]
    # yRange = [center[1] - crossSize, center[1] + crossSize]
    # imageSlice[center[1], xRange[0]:xRange[1]] = 32767
    # imageSlice[yRange[0]:yRange[1], center[0]] = 32767

    #draw box
    radius = np.int(np.rint(diameter / 2))
    xRange = [center[2] - radius, center[2] + radius]
    yRange = [center[1] - radius, center[1] + radius]
    imageSlice[yRange[0], xRange[0]:xRange[1]+1] = 32767
    imageSlice[yRange[1], xRange[0]:xRange[1]+1] = 32767
    imageSlice[yRange[0]:yRange[1]+1, xRange[0]] = 32767
    imageSlice[yRange[0]:yRange[1]+1, xRange[1]] = 32767

    return imageSlice

if __name__ == "__main__":
    serializer = NoduleSerializer("d:/project/tianchi/data/", "test")
    plotTrainSampleFromFile2D(serializer, "LKDS-00160", 4)