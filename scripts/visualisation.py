import matplotlib.pyplot as plt

resolution_dpi = 10


def plot_z_slice(show, image_path, z_slice, num_z_slices, image_stack, xscale, yscale, unit):
    print("Plotting z slice # {} of {}".format(z_slice, image_path))
    if unit == "CENTIMETER":
        xscale *= 10000
        yscale *= 10000
        unit = "$\mu$m"
    plt.figure(figsize=(6, 6))

    plt.title(f"Image #{image_path} Z-slice #{z_slice}")
    if num_z_slices != 1:
        image_slice = image_stack[z_slice]
    else:
        image_slice = image_stack
    x_extent = [0, image_slice.shape[1] * xscale]
    y_extent = [0, image_slice.shape[0] * yscale]
    plt.imshow(image_slice, cmap="Greens", extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]])
    plt.xlabel("x ({})".format(unit))
    plt.ylabel("y ({})".format(unit))

    plt.savefig(
        f"figures/slice_image/{image_path}-z{z_slice}.png", dpi=resolution_dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
