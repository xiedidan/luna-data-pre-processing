import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters
import scipy.ndimage

def segment_HU_scan(x):
    mask = np.asarray(x < -350, dtype='int32')
    for iz in range(mask.shape[0]):
        skimage.segmentation.clear_border(mask[iz], in_place=True)
        skimage.morphology.binary_opening(mask[iz], selem=skimage.morphology.disk(5), out=mask[iz])
        if np.sum(mask[iz]):
            mask[iz] = skimage.morphology.convex_hull_image(mask[iz])
    return mask

def segment_HU_scan_frederic(x, threshold=-350):
    mask = np.copy(x)
    binary_part = mask > threshold
    selem1 = skimage.morphology.disk(8)
    selem2 = skimage.morphology.disk(2)
    selem3 = skimage.morphology.disk(13)

    for iz in range(mask.shape[0]):
        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part[iz])  # fill body
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem1)
        mask[iz] *= filled_borders_mask


        mask[iz] = skimage.morphology.closing(mask[iz], selem2)
        mask[iz] = skimage.morphology.erosion(mask[iz], selem3)
        mask[iz] = mask[iz] < threshold

    return mask

def segment_HU_scan_elias(x, threshold=-350, pid='test', plot=False, verbose=False):
    mask = np.copy(x)
    binary_part = mask > threshold
    selem1 = skimage.morphology.disk(8)
    selem2 = skimage.morphology.disk(2)
    selem3 = skimage.morphology.disk(13)

    for iz in range(mask.shape[0]):
        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part[iz])  # fill body
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem1)
        mask[iz] *= filled_borders_mask

        mask[iz] = skimage.morphology.closing(mask[iz], selem2)
        mask[iz] = mask[iz] < threshold

    # params
    overlap_treshold = 7
    ratio_overlap_treshold = 0.015

    #discard disconnected regions, start at the middle slice and go to the head
    for iz in range(mask.shape[0]//2, mask.shape[0]-1): # use // for floor div
        overlap = mask[iz] * mask[iz+1] 
        label_image = skimage.measure.label(mask[iz+1])
        if verbose: 
            print('iz', iz)
        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:                
                total_overlap += overlap[coordinates[0], coordinates[1]]
            ratio_overlap = 1.*total_overlap/region.area
            if verbose:
                print('region', idx, ', t_overlap', total_overlap, ', r_overlap ', ratio_overlap, ', area ', region.area, ', center', np.round(region.centroid))
            if total_overlap < overlap_treshold or ratio_overlap < ratio_overlap_treshold:
                if verbose:
                    print('region', idx, 'in slice z=', iz-1, 'has a low overlap (', total_overlap, ratio_overlap, ') and will be discarded')
                for coordinates in region.coords: 
                    mask[iz+1, coordinates[0], coordinates[1]] = 0

    #discard disconnected regions, start at the middle slice and go to the head
    for iz in range(mask.shape[0]//2,0,-1 ):
        overlap = mask[iz] * mask[iz-1] 
        label_image = skimage.measure.label(mask[iz-1])
        if verbose: 
            print('iz', iz)
        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:                
                total_overlap += overlap[coordinates[0], coordinates[1]]
            ratio_overlap = 1.*total_overlap/region.area
            if verbose: 
                print('region', idx, ', t_overlap', total_overlap, ', r_overlap ', ratio_overlap, ', area ', region.area, ', center', np.round(region.centroid))
            if total_overlap < overlap_treshold or ratio_overlap < ratio_overlap_treshold:
                if verbose: 
                    print('region', idx, 'in slice z=', iz-1, 'has a low overlap (', total_overlap, ratio_overlap, ') and will be discarded')
                for coordinates in region.coords: 
                    mask[iz-1, coordinates[0], coordinates[1]] = 0


    #erode out the blood vessels and the borders of the lung for a cleaner mask
    for iz in range(mask.shape[0]):
        mask[iz] = skimage.morphology.binary_dilation(mask[iz], selem3)
        #mask[iz] = scipy.ndimage.binary_fill_holes(mask[iz])


    #if plot:
        # utils_plots.plot_all_slices(x, mask, pid, './plots/segment_HU_scan_elias/')

    return mask


def segment_HU_scan_ira(x, threshold=-350, min_area=300):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in range(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    # noise reduction
    mask = skimage.morphology.binary_opening(mask, skimage.morphology.cube(2))
    mask = np.asarray(mask, dtype='int8')

    # label regions
    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_label = sorted_regions[0].label
    lung_mask = np.asarray((label_image == lung_label), dtype='int8')

    # convex hull mask
    lung_mask_convex = np.zeros_like(lung_mask)
    for i in range(lung_mask.shape[2]):
        if np.any(lung_mask[:, :, i]):
            lung_mask_convex[:, :, i] = skimage.morphology.convex_hull_image(lung_mask[:, :, i])

    # old mask inside the convex hull
    mask *= lung_mask_convex
    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)

    for r in sorted_regions[1:]:
        if r.area > min_area:
            # make an image only containing that region
            label_image_r = label_image == r.label
            # grow the mask
            label_image_r = scipy.ndimage.binary_dilation(label_image_r,
                                                          structure=scipy.ndimage.generate_binary_structure(3, 2))
            # compute the overlap with true lungs
            overlap = label_image_r * lung_mask
            if not np.any(overlap):
                for i in range(label_image_r.shape[0]):
                    if np.any(label_image_r[i]):
                        label_image_r[i] = skimage.morphology.convex_hull_image(label_image_r[i])
                lung_mask_convex *= 1 - label_image_r

    return lung_mask_convex