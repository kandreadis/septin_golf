"""
This script contains all analysis and helper modules for this project.
Do not run this script separately, it is executed by main.py.
Author: Konstantinos Andreadis
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.signal import butter, filtfilt, find_peaks

from scripts.visualisation import plot_angular_profile, plot_filtered_signal, plot_grid_profile, plot_spacing_hist, \
    plot_extracted_image, plot_batch_analysis


def find_tif_files(folder_dir):
    all_files = os.listdir(os.path.normpath(sys.path[1] + "/" + folder_dir))
    tif_files = list(filter(lambda f: f.endswith('.tif'), all_files))
    num_images = len(tif_files)
    print("==== Found {} tif file(s) in {}! ====".format(num_images, folder_dir))

    images = []

    for tif_image_path in tif_files:
        image_path = os.path.normpath(sys.path[1] + "/" + folder_dir + "/" + tif_image_path)
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
            "full_path": os.path.normpath(folder_dir + "/" + tif_image_path)
        }
        images.append(image_extraction)
    return images


def threshold_img(img, thresh_low):
    print(" + Tresholding raw image...")
    _, img_thres = cv2.threshold(img, thresh_low, 255, cv2.THRESH_BINARY)
    return img_thres


def blur_img(img, gaussian_kernel_size):
    gaussian_sigma = 0
    img = cv2.GaussianBlur(img, ksize=(gaussian_kernel_size, gaussian_kernel_size), sigmaX=gaussian_sigma,
                           sigmaY=gaussian_sigma)
    return img


def extract_radial_profile(img, polar_grid_specs):
    max_radius = polar_grid_specs["max_radius"]
    min_radius = polar_grid_specs["min_radius"]
    radius_step = polar_grid_specs["radius_step"]
    angles_step = polar_grid_specs["angles_step"]
    center_x, center_y = polar_grid_specs["center_xy"]
    angles = np.arange(0, 360 + angles_step, angles_step)
    averaged_intensity_by_angle = np.zeros(len(angles))
    grid_xy = []
    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        x_unit_direct = np.cos(theta)
        y_unit_direct = np.sin(theta)
        x_coords = center_x + np.arange(min_radius, max_radius, radius_step) * x_unit_direct
        y_coords = center_y + np.arange(min_radius, max_radius, radius_step) * y_unit_direct
        x_coords = np.clip(x_coords, 0, img.shape[1] - 1).astype(int)
        y_coords = np.clip(y_coords, 0, img.shape[0] - 1).astype(int)
        intensity_values = img[y_coords, x_coords]
        averaged_intensity_by_angle[i] = np.average(intensity_values)
        grid_xy.append([x_coords, y_coords])
    return averaged_intensity_by_angle, angles, grid_xy


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def smooth_filter(array, spacing, low_freq, high_freq):
    fs = 1 / spacing
    frequencies = np.fft.fftfreq(len(array), 1 / fs)
    spectrum = np.fft.fft(array)
    filtered_array = bandpass_filter(array, low_freq, high_freq, fs)
    return filtered_array, frequencies, spectrum


def measure_angular_peaks(array, cutoff, radius, xscale, yscale):
    print(" + Detecting cross_section...")
    scale = np.sqrt(xscale ** 2 + yscale ** 2)
    peaks_locations, _ = find_peaks(array, height=cutoff)
    try:
        peaks_spacing = np.radians(np.gradient(peaks_locations)) * radius * scale
    except:
        peaks_spacing = []
    return peaks_spacing


def angular_spacing(img, img_params):
    print(" + Unfolding surfaceball...")
    img = threshold_img(img=img, thresh_low=3)
    polar_grid_specs = {
        "center_xy": [img.shape[0] // 2, img.shape[1] // 2],
        "min_radius": 0.75 * (img.shape[0] // 2),
        "max_radius": img.shape[0] // 2,
        "radius_step": 1,
        "angles_step": 1
    }
    avg_r = np.average([polar_grid_specs["min_radius"], polar_grid_specs["max_radius"]])
    angular_profile, angles, grid_xy = extract_radial_profile(img=img, polar_grid_specs=polar_grid_specs)
    plot_angular_profile(img=img, polar_grid_specs=polar_grid_specs, angular_profile=angular_profile,
                         angles=angles, xscale=img_params["xscale"], yscale=img_params["yscale"],
                         unit=img_params["unit"], grid_xy=None, path=img_params["path"])
    freq_bounds = [0.1, 0.2]
    smoothen_profile = smooth_filter(array=angular_profile, spacing=polar_grid_specs["angles_step"],
                                     high_freq=freq_bounds[1], low_freq=freq_bounds[0])
    filtered_angular_profile, frequencies, spectrum = smoothen_profile
    peaks_spacing = measure_angular_peaks(array=filtered_angular_profile, cutoff=3, radius=avg_r,
                                          xscale=img_params["xscale"], yscale=img_params["yscale"])
    plot_filtered_signal(array=angular_profile, frequencies=frequencies, spectrum=spectrum,
                         filtered_array=filtered_angular_profile, path=img_params["path"], freq_bounds=freq_bounds)
    return peaks_spacing


def measure_grid_peaks(matrix, xscale, yscale, path, unit):
    print(" + Detecting cross_section in grid...")
    # peaks_locations, _ = find_peaks(matrix.flatten(), height=255)
    # row_indices, col_indices = np.unravel_index(peaks_locations, matrix.shape)
    # peaks_locations = findpeaks(method="mask").fit(matrix)["Xdetect"].astype(int)
    kernel = np.ones((5, 5), np.uint8)
    matrix = cv2.dilate(matrix, kernel)
    contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    peak_indices_x = []
    peak_indices_y = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 1500:
            try:
                x, y = contour.mean(axis=0)[0]
            except:
                x, y = None, None
                print(" + Could not find any cross_section...")
            peak_indices_x.append(int(x))
            peak_indices_y.append(int(y))
    peak_indices_x = np.asarray(peak_indices_x)
    peak_indices_y = np.asarray(peak_indices_y)
    # peak_indices_x, peak_indices_y = np.where(matrix >= cutoff)
    peaks_locations = np.column_stack((peak_indices_x, peak_indices_y))
    plot_grid_profile(img=matrix, x=peak_indices_x, y=peak_indices_y, path=path, xscale=xscale, yscale=yscale,
                      unit=unit)
    radii = []
    for i in range(len(peaks_locations)):
        for j in range(len(peaks_locations)):
            if i != j:
                diff = peaks_locations[j] - peaks_locations[i]
                radius = np.sqrt(np.sum(np.square(diff))) * np.sqrt(xscale ** 2 + yscale ** 2)
                if radius <= 2:
                    radii.append(radius)
    peaks_spacing = np.asarray(radii)
    return peaks_spacing


def grid_spacing(img, img_params):
    print(" + Cropping surfaceball...")
    img = blur_img(img=img, gaussian_kernel_size=3)
    img = threshold_img(img=img, thresh_low=5)
    radius_cutoff = 0.75 * (img.shape[0] // 2)
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_center = np.sqrt((X - img.shape[0] // 2) ** 2 + (Y - img.shape[1] // 2) ** 2)
    img = np.where(dist_from_center <= radius_cutoff, img, 0)
    peaks_spacing = measure_grid_peaks(matrix=img, xscale=img_params["xscale"], yscale=img_params["yscale"],
                                       path=img_params["path"], unit=img_params["unit"])
    return peaks_spacing


def run(analysis_type):
    if analysis_type == "visualise_batch_results":
        surface_folder_dir = os.path.normpath("result/{}/".format("surface"))
        surface_dir = os.listdir(os.path.normpath(sys.path[1] + "/" + surface_folder_dir))
        surface_csv_files = list(filter(lambda f: f.endswith('.csv'), surface_dir))
        surface_csv_files = [os.path.normpath("result/surface/" + dir) for dir in surface_csv_files]
        cross_section_folder_dir = os.path.normpath("result/{}/".format("cross_section"))
        cross_section_dir = os.listdir(os.path.normpath(sys.path[1] + "/" + cross_section_folder_dir))
        cross_section_csv_files = list(filter(lambda f: f.endswith('.csv'), cross_section_dir))
        cross_section_csv_files = [os.path.normpath("result/cross_section/" + dir) for dir in cross_section_csv_files]
        all_csv_files = cross_section_csv_files + surface_csv_files
        num_images = len(all_csv_files)
        print("==== Found {} .csv file(s)! ====".format(num_images))
        print(" + Extracting batch results...")
        result_dict = {
            "path": [],
            "type": [],
            "spacing": [],
            "unit": []
        }
        for csv_file in all_csv_files:
            csv_content = pd.read_csv(os.path.normpath(sys.path[1] + "/" + csv_file), header=0).to_dict(
                orient="records")
            if len(csv_content) > 0:
                for spacing in csv_content:
                    result_dict["path"].append(spacing["path"])
                    result_dict["type"].append(spacing["type"])
                    result_dict["spacing"].append(spacing["spacing"])
                    result_dict["unit"].append(spacing["unit"])
        try:
            print(" + Saving batch result figure...")
            plot_batch_analysis(data=result_dict)
        except:
            pass
    else:
        tif_files = find_tif_files(os.path.normpath("data/" + analysis_type))
        for i in range(len(tif_files)):
            img_params = tif_files[i]
            img_raw = img_params["image_stack"]
            plot_extracted_image(img_params, os.path.normpath(analysis_type + "/" + img_params["path"]))
            print("# Analysing {}".format(img_params["path"]))
            if analysis_type == "surface":
                peaks_spacing = grid_spacing(img_raw, img_params)
            elif analysis_type == "cross_section":
                peaks_spacing = angular_spacing(img_raw, img_params)
            else:
                peaks_spacing = None
            img_file_path = os.path.normpath(img_params["folder_path"] + "/" + img_params["path"])
            result = {
                "path": img_file_path,
                "type": analysis_type,
                "spacing": peaks_spacing,
                "unit": img_params["unit"]
            }
            try:
                if len(peaks_spacing) > 0:
                    avg_spacing = np.average(peaks_spacing)
                    avg_spacing = round(avg_spacing, 3)
                else:
                    avg_spacing = "NONE"
            except:
                avg_spacing = "NONE"

            print(" + average ca. {} {}".format(avg_spacing, img_params["unit"]))
            plot_spacing_hist(peaks_spacing, img_params["unit"],
                              "average ca. {} {}".format(avg_spacing, img_params["unit"]),
                              path=os.path.normpath(analysis_type + "/" + img_params["path"]))
            np.set_printoptions(threshold=np.inf)
            df = pd.DataFrame.from_dict(result, orient="columns")
            df.to_csv(os.path.normpath("result/" + analysis_type + "/" + img_params["path"] + "_analysis.csv"),
                      index=False)
            print(" + Saved .csv, analysis complete!")
