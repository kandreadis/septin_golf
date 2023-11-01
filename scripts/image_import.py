import os
import sys

import numpy as np
import tifffile as tiff


def find_tif_files(folder_dir):
    all_files = os.listdir(sys.path[1] + "/" + folder_dir)
    tif_files = list(filter(lambda f: f.endswith('.tif'), all_files))
    num_images = len(tif_files)
    print("Found {} tif file(s) in {}!".format(num_images, folder_dir))

    images = []

    for tif_image_path in tif_files:
        image_path = sys.path[1] + "/" + folder_dir + "/" + tif_image_path
        image = tiff.imread(image_path)
        xscale, yscale, unit = None, None, None
        with tiff.TiffFile(image_path) as tif:
            for page in tif.pages:
                x_scale = page.tags["XResolution"].value
                y_scale = page.tags["YResolution"].value
                unit = str(page.tags["ResolutionUnit"].value)[8:]
                xscale = x_scale[1] / x_scale[0]
                yscale = y_scale[1] / y_scale[0]
        dimension = len(image.shape)
        if dimension == 3:
            num_z_slices = image.shape[0]
        else:
            num_z_slices = 1

        image_stack = np.asarray(image)
        if unit == "CENTIMETER":
            xscale *= 10000
            yscale *= 10000
            unit = "$\mu$m"
        if unit == "NONE":
            unit = "$\mu$m"
        image_extraction = {
            "path": tif_image_path,
            "num_z_slices": num_z_slices,
            "xscale": xscale,
            "yscale": yscale,
            "unit": unit,
            "image_stack": image_stack,
            "folder_path": folder_dir,
            "full_path": folder_dir + "/" + tif_image_path
        }
        images.append(image_extraction)
    return images
