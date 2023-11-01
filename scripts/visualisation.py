import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

resolution_dpi = 300


def plot_extracted_image(tif_files, path):
    if type(tif_files) == dict:
        tif_files = [tif_files]
    for tif_image in tif_files:
        num_z_slice = tif_image["num_z_slices"]
        if num_z_slice == 1:
            z_slice = 0
            plot_z_slice(image_path=tif_image["path"], image_stack=tif_image["image_stack"],
                         xscale=tif_image["xscale"], yscale=tif_image["yscale"], unit=tif_image["unit"],
                         z_slice=z_slice, num_z_slices=num_z_slice, path=path)
        else:
            for z_slice in range(num_z_slice):
                plot_z_slice(image_path=tif_image["path"], image_stack=tif_image["image_stack"],
                             xscale=tif_image["xscale"], yscale=tif_image["yscale"], unit=tif_image["unit"],
                             z_slice=z_slice, num_z_slices=num_z_slice, path=path)


def plot_z_slice(image_path, z_slice, num_z_slices, image_stack, xscale, yscale, unit, path):
    plt.figure(figsize=(6, 6))

    plt.title(f"Image #{image_path} Z-slice #{z_slice}")
    if num_z_slices != 1:
        image_slice = image_stack[z_slice]
    else:
        image_slice = image_stack
    x_extent = [0, image_slice.shape[0] * xscale]
    y_extent = [0, image_slice.shape[1] * yscale]
    plt.imshow(image_slice, cmap="Greens", extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
    plt.xlabel("x ({})".format(unit))
    plt.ylabel("y ({})".format(unit))
    plt.savefig(f"figures/{path}.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.close()


def plot_angular_profile(img, polar_grid_specs, angular_profile, angles, xscale, yscale, unit, grid_xy, path):
    center_x, center_y = polar_grid_specs["center_xy"]
    center_x *= xscale
    center_y *= yscale
    min_radius = polar_grid_specs["min_radius"] * xscale
    max_radius = polar_grid_specs["max_radius"] * xscale
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].set_title("Image")
    x_extent = [0, img.shape[0] * xscale]
    y_extent = [0, img.shape[1] * yscale]
    ax[0].imshow(img, cmap='gray', extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
    ax[0].set_xlabel("x ({})".format(unit))
    ax[0].set_ylabel("y ({})".format(unit))
    ax[0].plot(center_x, center_y, "rx")
    circle_min = patches.Circle((center_x, center_y), min_radius, fill=False, color='red',
                                linewidth=2)
    ax[0].add_patch(circle_min)
    circle_max = patches.Circle((center_x, center_y), max_radius, fill=False, color='red',
                                linewidth=2)
    ax[0].add_patch(circle_max)

    if grid_xy is not None:
        for grid in grid_xy:
            ax[0].scatter(grid[0] * xscale, grid[1] * yscale, marker=".", s=1, color="white")

    ax[1].set_title("Averaged angular intensity profile")
    ax[1].plot(angles, angular_profile)
    ax[1].set_ylabel("Intensity")
    ax[1].set_xlabel("Angle (deg)")
    plt.tight_layout()
    plt.savefig(f"figures/spikes/{path}_angular-profile.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.close()


def plot_filtered_signal(array, frequencies, spectrum, freq_bounds, filtered_array, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(array)
    ax[0].set_title('Original Angular Profile')
    ax[0].set_xlabel('$\Theta$ (degrees)')
    ax[0].set_ylabel('Amplitude')
    ax[1].set_title('Frequency Spectrum of Original Angular Profile')
    ax[1].plot(frequencies[frequencies >= 0], np.abs(spectrum)[frequencies >= 0])
    ax[1].axvline(freq_bounds[0], color="red", linestyle="--", label="bandpass_low")
    ax[1].axvline(freq_bounds[1], color="red", linestyle="--", label="bandpass_high")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].legend()
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[2].plot(filtered_array)
    ax[2].set_title('Band-Pass: Filtered Angular Profile')
    ax[2].set_xlabel('$\Theta$ (degrees)')
    ax[2].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(f"figures/spikes/{path}_filtered_angular-profile.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.close()


def plot_grid_profile(img, x, y, xscale, yscale, unit, path):
    x_extent = [0, img.shape[1] * xscale]
    y_extent = [0, img.shape[0] * yscale]
    plt.figure()
    plt.imshow(img, alpha=0.5, extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]])
    plt.scatter(x * xscale, y * yscale, color="blue", s=5, marker="x")
    plt.xlabel("x ({})".format(unit))
    plt.ylabel("y ({})".format(unit))
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig(f"figures/golf/{path}_grid.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.close()


def plot_spacing_hist(array, unit, label, path):
    print(label)
    plt.figure()
    plt.hist(array, label=label)
    plt.xlabel("spacing {}".format(unit))
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"figures/{path}_spacings.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.close()
