# geometry_utils.py
# ----------------------------------------------------------------
# Utility functions for computing detector V-range and sub-geometry
# from TIGRE geometry objects.
#
# These functions help reduce reconstruction volume and detector area
# by computing physical projection ranges for a given Z slice range.
#
# Functions:
# - compute_ab_vrange_corrected(...)
# - compute_subgeo_for_zrange(...)
# - trim_geo_after_padding (...) is used for calcuation mask after padding 
# - bin_projection_geo (...) is used for COR calcuation for golden search
#
# Author: Kubra Kumrular (RKK)
# ----------------------------------------------------------------

import numpy as np
import copy


def compute_ab_vrange_corrected(geo, z_start, z_end, angles=None):

    """
    Computes the vertical detector pixel range (a, b) corresponding to a selected Z range.

    Parameters:
    - geo: TIGRE geometry object
    - z_start, z_end: Slice indices (voxel space)
    - angles: Optional list of angles in degrees (default = [135, 315])

    Returns:
    - a, b: Detector row indices to be used
    """
    # Scalable FBP decomposition for cone-beam CT reconstruction
    # https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8O9RNFEAAAAJ&sortby=pubdate&citation_for_view=8O9RNFEAAAAJ:p2g8aNsByqUC
    nz, ny, nx = geo.nVoxel
    dz, dy, dx = geo.dVoxel
    phys_z_min = (z_start - nz / 2) * dz
    phys_z_max = (z_end - 1 - nz / 2) * dz

    # Volume corners in physical coordinates (X,Y,Z)
    corners = [
        [-geo.sVoxel[2]/2, -geo.sVoxel[1]/2, phys_z_min],
        [-geo.sVoxel[2]/2,  geo.sVoxel[1]/2, phys_z_min],
        [ geo.sVoxel[2]/2, -geo.sVoxel[1]/2, phys_z_min],
        [ geo.sVoxel[2]/2,  geo.sVoxel[1]/2, phys_z_min],
        [-geo.sVoxel[2]/2, -geo.sVoxel[1]/2, phys_z_max],
        [-geo.sVoxel[2]/2,  geo.sVoxel[1]/2, phys_z_max],
        [ geo.sVoxel[2]/2, -geo.sVoxel[1]/2, phys_z_max],
        [ geo.sVoxel[2]/2,  geo.sVoxel[1]/2, phys_z_max]
    ]

    def project_to_detector(pt, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Source position (TIGRE coordinate system)
        S = np.array([
            0,                          # X source position (iso-center)
            -geo.DSO,                   # Y source position (negative Y axis)
            0                           # Z source position
        ])

        # Rotate point around Z-axis (rotation axis)
        rot_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        pt_rot = rot_mat @ pt

        # Ray direction
        ray = pt_rot - S
        ray = ray / np.linalg.norm(ray)

        # Find intersection with detector plane (Y = DSD - DSO)
        t = (geo.DSD - geo.DSO - S[1]) / ray[1]
        det_hit = S + t * ray

        # Convert to detector coordinates (V,U)
        # V-coordinate is along Z-axis
        v_mm = det_hit[2] - geo.offDetector[0]
        v_px = v_mm / geo.dDetector[0] + geo.nDetector[0]/2
        return v_px

    if angles is None:
        angles = [135, 315]

    all_v = []
    for angle in angles:
        for corner in corners:
            v_px = project_to_detector(corner, angle)
            all_v.append(v_px)

    a = max(0, int(np.floor(min(all_v))))
    b = min(geo.nDetector[0] - 1, int(np.ceil(max(all_v))))
    return a, b

def compute_subgeo_for_zrange(geo, z_start, z_end, v_start, v_end):
    """
    Returns a copy of the geometry object for a smaller region in Z and V.

    Parameters:
    - geo: TIGRE geometry object
    - z_start, z_end: Voxel slice range
    - v_start, v_end: Detector row range (vertical)

    Returns:
    - geo_sub: Modified TIGRE geometry object
    """
    geo_sub = copy.deepcopy(geo)
    #=== VOXEL (IMAGE) SECTION ===
    nSlices = z_end - z_start
    geo_sub.nVoxel[0] = nSlices
    geo_sub.sVoxel[0] = geo_sub.nVoxel[0] * geo.dVoxel[0]
    phys_z_center = (z_start - geo.nVoxel[0]/2 + nSlices/2) * geo.dVoxel[0]
    geo_sub.offOrigin = np.array([phys_z_center, 0, 0])
    # === DETECTOR SECTION ===
    v_height = v_end - v_start
    geo_sub.nDetector = np.array([v_height, geo.nDetector[1]])
    geo_sub.sDetector = geo_sub.nDetector * geo_sub.dDetector
    v_center_px = v_start + v_height / 2
    off_detector_v = (v_center_px - geo.nDetector[0] / 2) * geo.dDetector[0]
    geo_sub.offDetector = np.array([off_detector_v, 0])

    return geo_sub



def trim_geo_after_padding(geo, pad_width):
    geo = copy.deepcopy(geo)

    geo.nVoxel[2] -= 2 * pad_width  # X (U axis)
    geo.sVoxel[2] = geo.nVoxel[2] * geo.dVoxel[2]

    geo.nVoxel[1] -= 2 * pad_width  # Y (V axis)
    geo.sVoxel[1] = geo.nVoxel[1] * geo.dVoxel[1]

    geo.nDetector[1] -= 2 * pad_width  # only X axis in detector 
    geo.sDetector[1] = geo.nDetector[1] * geo.dDetector[1]

    return geo



def bin_projection_geo(proj,geo, bin_factor):
    shape = proj.shape
    width = proj.shape[2]
    half_panel = (width - 1)/2
    new_u = shape[2] // bin_factor
    proj_binned = proj[:, :, :new_u * bin_factor].reshape(shape[0], shape[1], new_u, bin_factor)
    # Create binned geometry
    geo_binned = copy.deepcopy(geo)
    geo_binned.nDetector[1] = proj_binned.shape[2]
    geo_binned.dDetector[1] = geo.dDetector[1] * bin_factor
    geo_binned.sDetector[1] = geo_binned.nDetector[1] * geo_binned.dDetector[1]

    # (Optional but recommended) Update voxel size and image grid accordingly:
    geo_binned.nVoxel[1] = geo.nVoxel[1] // bin_factor
    geo_binned.nVoxel[2] = geo.nVoxel[2] // bin_factor
    geo_binned.dVoxel[1] = geo.dVoxel[1] * bin_factor
    geo_binned.dVoxel[2] = geo.dVoxel[2] * bin_factor
    geo_binned.sVoxel[1] = geo_binned.nVoxel[1] * geo_binned.dVoxel[1]
    geo_binned.sVoxel[2] = geo_binned.nVoxel[2] * geo_binned.dVoxel[2]

    print(geo_binned)
    return proj_binned.mean(axis=-1),geo_binned