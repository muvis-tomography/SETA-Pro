
#center_of_rotation_calculation.py

#This function estimates the Center of Rotation (COR) offset for CT reconstruction
#based on a central projection slice using a combination of coarse grid search and
#refined golden section search.

#Parameters:
#    proj_central (np.ndarray): Central projection slice (shape: [angles, 1, width])
#    geo2D (tigre geometry): TIGRE 2D geometry object
#    angles (np.ndarray): Array of angles in radians
#    gpuids (list or int): GPU device ID(s)
#    cor_mode (int): 0 = auto detection, 1 = manual entry
#    manual_cor_value (float): COR offset in mm (used only if cor_mode = 1)

#Returns:
#    float: Estimated or manually assigned COR offset in millimeters

# Author: Kubra Kumrular (RKK)

##% Define the libraries 

import numpy as np
import matplotlib.pyplot as plt
from tigre.utilities import gpu
import copy
from scipy.ndimage import sobel, gaussian_filter
import math

import tigre.algorithms as algs
from .geometry_utils import bin_projection_geo

def calculate_COR(proj_central, geo2D, angles, gpuids, cor_mode=0, manual_cor_value=0.0):
    """
    Calculate optimal COR (Center of Rotation) offset in mm.
    
    Parameters:
        proj_central (ndarray): Central slice projection (N,1,U)
        geo2D (Tigre Geometry): 2D TIGRE geometry
        angles (ndarray): Angles in radians
        gpuids (int or list): GPU device id(s)
        cor_mode (int): 0 = auto, 1 = manual
        manual_cor_value (float): If manual selected, this value is used
        
    Returns:
        float: Best COR offset in mm
    """
    geo = copy.deepcopy(geo2D)

    if cor_mode == 1:
        geo.COR = manual_cor_value
        proj_central = None  
        print(f"Using user-provided COR: {geo.COR:.3f} mm")
        return geo.COR
    
    # --- STEP 1: Determine Binning Parameters ---
    detector_width_px = proj_central.shape[2] 
    voxel_size_mm = geo.dVoxel[2] 
    #grid_range_mm = (detector_width_px * voxel_size_mm) / 4
    grid_range_mm = min((detector_width_px * voxel_size_mm) / 4, 5.0)

    initial_binning = int(np.ceil(detector_width_px / 128))
    initial_binning = min(initial_binning, 16)
    #n_grid = n_grid = min(int(4 * (grid_range_mm / voxel_size_mm) + 1), 101)
    #n_grid = int(4 * (grid_range_mm / (voxel_size_mm * initial_binning)) + 1)
    #n_grid = int(n_grid)
    #print(n_grid)

       # --- STEP 2: Preprocess projection data ---
    # Apply Gaussian blur (optional but improves robustness)
    #proj_blurred = gaussian_filter(proj_central, sigma=[0, 0, 0.5])  # axis=2 = detector axis (U)
    sigma_u = initial_binning / 2
    #sigma_u= 0.5
    proj_processed = gaussian_filter(proj_central, sigma=[0, 0, sigma_u]) ## Smooth along detector axis (U)
    # Apply Sobel filter for edge enhancement
    proj_processed = sobel(proj_processed, axis=2)
    proj_filtered_binned,geo_binned = bin_projection_geo(proj_processed,geo, bin_factor=initial_binning)

    # --- STEP 3: Coarse grid search using binned projections ---
    #grid_range_mm = 2.0  # total +/- range
    #n_grid = 41
    step_mm = voxel_size_mm * initial_binning
    n_grid = int(4 * (grid_range_mm / step_mm) + 1)
    if n_grid % 2 == 0:
        n_grid += 1  # odd number, middle

    offsets = np.linspace(-grid_range_mm, grid_range_mm, n_grid)
    sharpness_vals = []
    #print(proj_filtered_binned)

    print("Running initial grid search...")
    #print(geo_binned)

    for offset in offsets:
        #geo.rotation_axis_offset = offset
        voxel_mm_binnned = geo_binned.dVoxel[2]
        geo_binned.COR= offset * voxel_mm_binnned
        print("binned", geo_binned.COR)
        print(geo_binned)
        reco = algs.fdk(proj_filtered_binned, geo_binned, angles, gpuids=gpuids)
        sharpness = np.sum(reco[0] ** 2)
        sharpness_vals.append(sharpness)

    best_idx = np.argmax(sharpness_vals)
    if 0 < best_idx < len(offsets) - 1:
        x = offsets[best_idx - 1 : best_idx + 2]
        y = sharpness_vals[best_idx - 1 : best_idx + 2]
        z = np.polyfit(x, y, 2)
        coarse_offset = -z[1] / (2 * z[0])
  
    else:
        coarse_offset = offsets[best_idx]

    scale = geo_binned.dDetector[1] / geo.dDetector[1]  
    print("scale",scale)
    coarse_offset_corrected = coarse_offset #* geo_binned.dVoxel[2]
    print("coarse_offset",np.abs(coarse_offset_corrected) )

    # --- STEP 4: Refined Golden Section Search using full-resolution data ---
    def sharpness_metric(offset):
        voxel_mm = geo.dVoxel[2]
        geo.COR  = offset * voxel_mm
        print(geo.COR)
        reco = algs.fdk(proj_processed, geo, angles, gpuids=gpuids)
        return np.sum(reco[0] ** 2)

    def golden_section_search(f, a, b, tol=0.001, max_iter=100):
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        eval_c = f(c)
        eval_d = f(d)
        for _ in range(max_iter):
            if abs(c - d) < tol:
                break
            if eval_c > eval_d:
                b = d
                d = c
                eval_d = eval_c
                c = b - (b - a) / gr
                eval_c = f(c)
            else:
                a = c
                c = d
                eval_c = eval_d
                d = a + (b - a) / gr
                eval_d = f(d)

    # Fit parabola to last three points (c, mid, d)
        x_fit = [a, (a + b) / 2, b]
        y_fit = [f(x) for x in x_fit]
        z = np.polyfit(x_fit, y_fit, 2)
        min_point = -z[1] / (2 * z[0])
        return min_point

    print("Running refined search (GSS)...")
    voxel_mm = geo.dVoxel[2]
    search_a = max(0.0, coarse_offset_corrected - 2 * voxel_mm)
    search_b = coarse_offset_corrected + 2 * voxel_mm
    refined_offset = golden_section_search(sharpness_metric, search_a, search_b, tol=0.0001 * voxel_mm)
    refined_offset = np.abs(refined_offset) * voxel_mm

    print(f"[COR] Estimated optimal COR offset = {refined_offset:.6f} mm")
    return refined_offset

    #refined_offset = golden_section_search(sharpness_metric,
                                           #coarse_center_corrected - 2,
                                          # coarse_center_corrected + 2,
                                           #tol = 0.001 * voxel_size_mm)
    #print(refined_offset)
    #refined_offset = np.abs(float(refined_offset ))#*  geo.dVoxel[1] )) #*  geo.dVoxel[1]
    #print(f"Estimated optimal COR offset: {refined_offset:.5f} mm")
    #det_center = geo.sDetector[1] / 2
    #rotation_axis_offset = refined_offset - det_center
    #print(f"Final COR offset to apply in TIGRE: {rotation_axis_offset:.5f} mm")
    #return rotation_axis_offset
    #return refined_offset 