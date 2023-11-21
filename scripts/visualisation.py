"""
This script contains all visualisation modules used by this project.
Do not run this script separately, it is executed by analysis.py.
Author: Konstantinos Andreadis
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

resolution_dpi = 200


def axis_scalebar(ax, unit, dx):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(2)
    try:
        if unit == "$\mu$m":
            unit = "um"
        elif unit == "px":
            dx = 1
        ax.add_artist(
            ScaleBar(1, unit, length_fraction=2 * dx, location="upper right", border_pad=0.5, box_color="k", color="w",
                     box_alpha=0.7))
    except:
        pass


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
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Image #{image_path} Z-slice #{z_slice}")
    if num_z_slices != 1:
        image_slice = image_stack[z_slice]
    else:
        image_slice = image_stack
    x_extent = [0, image_slice.shape[0] * xscale]
    y_extent = [0, image_slice.shape[1] * yscale]
    ax.imshow(image_slice, cmap="RdPu", extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
    sm = plt.cm.ScalarMappable(cmap="RdPu")
    sm.set_array(image_slice)
    axis_scalebar(ax, dx=xscale, unit=unit)
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Intensity (a.u.)")
    plt.savefig(f"figures/{path}.png", dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
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
    ax[0].plot(center_x, center_y, "rx")
    axis_scalebar(ax=ax[0], unit=unit, dx=xscale)
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
    ax[1].set_ylabel("Intensity (a.u.)")
    ax[1].set_xlabel("Angle (deg)")
    plt.tight_layout()
    plt.savefig(f"figures/cross_section/{path}_angular-profile.png", dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
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
    plt.savefig(f"figures/cross_section/{path}_filtered_angular-profile.png", dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_grid_profile(img, x, y, xscale, yscale, unit, path):
    x_extent = [0, img.shape[1] * xscale]
    y_extent = [0, img.shape[0] * yscale]
    fig, ax = plt.subplots()
    ax.imshow(img, alpha=0.5, extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]], cmap="RdPu")
    ax.scatter(x * xscale, y * yscale, color="blue", s=5, marker="x")
    ax.set_aspect('equal')
    ax.set_title("Detected cross_section in circular mask")
    axis_scalebar(ax=ax, unit=unit, dx=xscale)
    fig.tight_layout()
    plt.savefig(f"figures/surface/{path}_grid.png", dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_spacing_hist(array, unit, label, path):
    plt.figure()
    plt.hist(array, label=label)
    plt.xlabel("spacing ({})".format(unit))
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"figures/{path}_spacings.png", dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_batch_analysis(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    unit = data["unit"][0]
    sns.boxplot(ax=ax, y=data["spacing"], hue=data["type"], legend="brief", medianprops={"color": "w", "linewidth": 2})
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylabel("Spike spacing ({})".format(unit))

    fig.tight_layout()
    plt.savefig("figures/batch_analysis/batch_analysis.png", dpi=resolution_dpi, bbox_inches='tight')
    plt.show()
    # plt.close()
