import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from scripts.image_import import find_tif_files
from scripts.visualisation import plot_angular_profile, plot_filtered_signal, plot_grid_profile, plot_spacing_hist, \
    plot_extracted_image


def threshold_img(img, thresh_low):
    print("Tresholding raw image...")
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


def analyse_angular_peaks(array, cutoff, radius, xscale, yscale):
    print("Detecting spikes...")
    scale = np.sqrt(xscale ** 2 + yscale ** 2)
    peaks_locations, _ = find_peaks(array, height=cutoff)
    try:
        peaks_spacing = np.radians(np.gradient(peaks_locations)) * radius * scale
    except:
        peaks_spacing = []
    return peaks_spacing


def angular_spacing(img, img_params):
    print("Unfolding golfball...")
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
    peaks_spacing = analyse_angular_peaks(array=filtered_angular_profile, cutoff=3, radius=avg_r,
                                          xscale=img_params["xscale"], yscale=img_params["yscale"])
    plot_filtered_signal(array=angular_profile, frequencies=frequencies, spectrum=spectrum,
                         filtered_array=filtered_angular_profile, path=img_params["path"], freq_bounds=freq_bounds)
    return peaks_spacing


def analyse_grid_peaks(matrix, xscale, yscale, path, unit):
    print("Detecting spikes in grid...")
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
            x, y = contour.mean(axis=0)[0]
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
    # radii = np.sqrt(
    #     (peak_indices_x - matrix.shape[0] // 2) ** 2 + (peak_indices_y - matrix.shape[1] // 2) ** 2) *
    peaks_spacing = np.asarray(radii)
    # peaks_spacing = peaks_spacing[peaks_spacing > 0.1]                      np.max(peaks_spacing), unit))
    return peaks_spacing


def grid_spacing(img, img_params):
    print("Cropping golfball...")
    img = blur_img(img=img, gaussian_kernel_size=3)
    img = threshold_img(img=img, thresh_low=5)
    radius_cutoff = 0.75 * (img.shape[0] // 2)
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_center = np.sqrt((X - img.shape[0] // 2) ** 2 + (Y - img.shape[1] // 2) ** 2)
    img = np.where(dist_from_center <= radius_cutoff, img, 0)
    peaks_spacing = analyse_grid_peaks(matrix=img, xscale=img_params["xscale"],
                                       yscale=img_params["yscale"], path=img_params["path"], unit=img_params["unit"])
    return peaks_spacing


def run(analysis_type):
    tif_files = find_tif_files("data/" + analysis_type)
    for i in range(len(tif_files)):
        img_params = tif_files[i]
        img_raw = img_params["image_stack"]
        plot_extracted_image(img_params, analysis_type + "/" + img_params["path"])
        print("================")
        print("Analysing {}".format(img_params["path"]))
        if analysis_type == "golf":
            peaks_spacing = grid_spacing(img_raw, img_params)
        elif analysis_type == "spikes":
            peaks_spacing = angular_spacing(img_raw, img_params)
        else:
            peaks_spacing = None
        img_file_path = img_params["folder_path"] + "/" + img_params["path"]
        result = {
            "path": img_file_path,
            "type": analysis_type,
            "spacing": peaks_spacing,
            "unit": img_params["unit"]
        }
        try:
            avg_spacing = np.average(peaks_spacing)
            avg_spacing = round(avg_spacing, 3)
        except:
            avg_spacing = "UNKNOWN"
        plot_spacing_hist(peaks_spacing, img_params["unit"],
                          "average ca. {} {}".format(avg_spacing, img_params["unit"]),
                          path=analysis_type + "/" + img_params["path"])
        np.set_printoptions(threshold=np.inf)
        df = pd.DataFrame.from_dict(result, orient="columns")
        df.to_csv("result/" + analysis_type + "/" + img_params["path"] + "_analysis.csv", index=False)
        print("Saved .csv, analysis complete!")
