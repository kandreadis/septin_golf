from scripts.image_import import find_tif_files
from scripts.visualisation import plot_z_slice


def extract_image(tif_files):
    if type(tif_files) == dict:
        tif_files = [tif_files]
    for tif_image in tif_files:
        print(tif_image)
        num_z_slice = tif_image["num_z_slices"]
        if num_z_slice == 1:
            z_slice = 0
            plot_z_slice(show=True, image_path=tif_image["path"], image_stack=tif_image["image_stack"],
                         xscale=tif_image["xscale"], yscale=tif_image["yscale"], unit=tif_image["unit"],
                         z_slice=z_slice, num_z_slices=num_z_slice)
        else:
            for z_slice in range(num_z_slice):
                plot_z_slice(show=True, image_path=tif_image["path"], image_stack=tif_image["image_stack"],
                             xscale=tif_image["xscale"], yscale=tif_image["yscale"], unit=tif_image["unit"],
                             z_slice=z_slice, num_z_slices=num_z_slice)


def run():
    tif_files = find_tif_files("data/20231031_examples")
    spike_file = tif_files[-1]
    golfball_file = tif_files[2]
    extract_image(spike_file)
