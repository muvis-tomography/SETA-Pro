# main_reconstruction.py
# ------------------------------------------------------------
# Main Reconstruction Pipeline for CT Data (Diondo & Nikon Systems)
#
# This module defines the main reconstruction flow used to:
# - Estimate Center of Rotation (COR)
# - Configure TIGRE geometry
# - Apply padding and sub-geometry cropping (Z-range)
# - Run reconstruction based on user-selected mode (2D, full 3D, or Z-range 3D)
# - Apply post-processing such as circular masking
#
# Supported modes:
#   Mode 1 = Central slice (2D)
#   Mode 2 = Full 3D volume reconstruction (chunked)
#   Mode 3 = Z-range-limited 3D reconstruction
#
# This module assumes that user inputs are collected via a separate GUI
# (see: user_interface.py) and geometry is read using DiondoGeometryReader or NikonGeometryReader.
#
# Depends on:
# - geometry_reader.py
# - load_projections.py
# - pre_processing.py
# - post_processing.py
# - logger.py
#
# Author: Kubra Kumrular (RKK)
# ------------------------------------------------------------

import numpy as np 
import glob
import os
import tigre.algorithms as algs
from PIL import Image
import matplotlib.pyplot as plt
from tigre.utilities import gpu
import time
import xml.etree.ElementTree as ET
import gc

from .logger             import write_log, log_step
from .pre_processing     import process_projection
from .geometry_utils     import compute_ab_vrange_corrected, compute_subgeo_for_zrange, trim_geo_after_padding,bin_projection_geo
from .geometry_reader    import DiondoGeometryReader, NikonGeometryReader
from .user_interface     import get_user_inputs
from .post_processing    import apply_circular_mask_inplace_parallel
from .center_of_rotation_calculation      import calculate_COR

def main_reconstruction_flow(mode, proj_data, geo_reader, angles, chunk_generator=None,
                             gpuids=[0], z_start=None, z_end=None,
                             cor_mode=0, cor_value= None,
                             apply_mask=False, output_dir=None, pad_width=0, geo=None, 
                             raw_shape=None, user_inputs=None):
    """
    Main reconstruction pipeline:
    - Calculates COR from 2D central slice
    - Updates geo object with COR
    - Calls reconstruction based on mode
    - Applies circular mask if requested
    - Saves output
    
    Parameters:
        mode: int (1, 2, 3)
        proj_data: np.ndarray
        geo_reader: DiondoGeometryReader
        angles: ndarray
        chunk_generator: generator (only for mode 2)
        gpuids: list of GPU IDs
        z_start, z_end: int (for mode 3)
        cor_mode: int (0=auto, 1=manual)
        manual_cor_value: float
        apply_mask: bool
        output_dir: str
    """
    # === 1. COR calculation===
 
    if cor_mode == 0:  # auto
        with log_step("COR Estimation"):
        # raw shape ve path'lerden tek bir sinogram yükle
            center_row = raw_shape[0] // 2 if raw_shape is not None else None
            if user_inputs["reader_type"] == 0:
                file_list = sorted(glob.glob(os.path.join(user_inputs["proj_folder"], '*.raw')))
                file_list = file_list[:geo_reader.get_projection_count()]
                proj_central = np.stack([
                np.fromfile(f, dtype=np.uint16).reshape(raw_shape)[center_row:center_row+1, :]
                for f in file_list
            ])
            if user_inputs["reader_type"] == 1:
                scan_folder = os.path.dirname(user_inputs["xml_file"])
                #file_list = sorted(glob.glob(os.path.join(scan_folder, '*.tif')))
                #file_list = sorted(glob.glob(glob.escape(os.path.join(scan_folder, '*.tif'))))
                search_pattern = os.path.join(glob.escape(scan_folder), '*.tif')
                file_list = sorted(glob.glob(search_pattern))
                file_list = file_list[:geo_reader.get_projection_count()]
                proj_central = np.stack([
                np.array(Image.open(f))[np.array(Image.open(f)).shape[0] // 2 : np.array(Image.open(f)).shape[0] // 2 + 1, :]
                for f in file_list
            ])
        
            #proj_central = np.stack([
            #    np.fromfile(f, dtype=np.uint16).reshape(raw_shape)[center_row:center_row+1, :]
            #    for f in file_list
            #])
            proj_central = -np.log(np.clip(proj_central, 1e-6, None) / 60000.0).astype(np.float32)

            geo2D = geo_reader.get_geometry("2D")
            cor_value = calculate_COR(proj_central, geo2D, angles, gpuids, cor_mode=0)
            write_log(f"AUTO COR used. Estimated COR: {cor_value:.3f} mm")
    else:
            #cor_value = user_inputs["cor_value"]
        cor_value  = float(cor_value)
        proj_central = None
        write_log(f"Using manual COR: {cor_value:.3f} mm")

    # === 2. GEO arrangement ===
    if mode == 1:
        geo = geo_reader.get_geometry("2D")
        geo.COR = np.round(cor_value, 3)
        if pad_width is not None and pad_width > 0:
            geo.nDetector[1] += 2 * pad_width
            geo.sDetector[1] = geo.nDetector[1] * geo.dDetector[1]
            #geo.nVoxel[0] += 2 * pad_width  # zy diraction
            geo.nVoxel[1] += 2 * pad_width  # X diraction
            geo.nVoxel[2] += 2 * pad_width  # X diraction
            geo.sVoxel[2] = geo.nVoxel[2] * geo.dVoxel[2]
            geo.sVoxel[1] = geo.nVoxel[1] * geo.dVoxel[1]

            #write_log(f"Detector geometry updated for padding: New width = {geo.nDetector[1]}")
            #write_log(f"2D - Geometry: \n{geo}")
        write_log(f"2D - Geometry: \n{geo}")

    elif geo is None:
        geo_full = geo_reader.get_geometry("3D")
        geo_full.COR = np.round(cor_value, 3)

        if mode == 3:
            a, b = compute_ab_vrange_corrected(geo_full, z_start, z_end, angles)
            geo = compute_subgeo_for_zrange(geo_full, z_start, z_end, a, b)

        elif mode ==2:
            geo = geo_full

    geo.COR = np.round(cor_value, 3)

    #print(pad_width)

        # === PADDING APPLY TO FINAL geo ===
    if pad_width is not None and pad_width > 0 and mode != 1:
        if mode == 2:
            geo = geo_reader.get_geometry("3D")
            geo.COR = np.round(cor_value, 3)
        geo.nDetector[1] += 2 * pad_width
        geo.sDetector[1] = geo.nDetector[1] * geo.dDetector[1]
        geo.nVoxel[2] += 2 * pad_width
        geo.nVoxel[1] += 2 * pad_width
        geo.sVoxel[2] = geo.nVoxel[2] * geo.dVoxel[2]
        geo.sVoxel[1] = geo.nVoxel[1] * geo.dVoxel[1]
            #write_log("Padding applied to geometry.")

    if mode == 3:
        write_log(f"3D sub - Geometry: \n{geo}")
    if mode == 2:
        write_log(f"3D Geometry: \n{geo}")

    # === 3. Reconstruction ===
    if mode == 1:
        #print("\n Mode 1:",geo)
        print("\n Mode 1: Central Slice")        
        recon = reconstruct2D(proj_data, geo, angles, gpuids)
    elif mode == 2:
        #print("\n Mode 2:",geo)
        print("\n Mode 2: Full Volume Chunked")
        recon = reconstruct_chunked(chunk_generator, geo, gpuids)        
    elif mode == 3:
        #print("\n Mode 3:",geo)
        print("\n Mode 3: Z-range")
        recon = reconstruct3D(proj_data, geo, angles, gpuids)     
    else:
        raise ValueError("Invalid mode")
    # === Padding Crop ===
    if pad_width is not None and pad_width > 0:
        if recon.ndim == 3:
            recon = recon[:, pad_width:-pad_width, pad_width:-pad_width]
        elif recon.ndim == 2:
            recon = recon[pad_width:-pad_width, pad_width:-pad_width]
        #write_log(f"Padding applied: Yes ({pad_width} pixels on each side)")
           # Geo'yu cropla (mode zaten int)
        geo = trim_geo_after_padding(geo, pad_width)
    #else:
        #write_log("Padding applied: No")
    
 
    # === Masking (apply crop volume) ===
    if apply_mask:
        t0 = time.time()
        recon = apply_circular_mask_inplace_parallel(recon, geo)
        elapsed = time.time() - t0
        write_log(f"Masking applied: Yes (took {elapsed:.2f} sec)")
    else:
        write_log("Masking applied: No")


    return recon, proj_central, cor_value



def reconstruct3D(proj_data, geo, angles, gpuids):
    return algs.fdk(proj_data, geo, angles, gpuids=gpuids,dowang = False)

def reconstruct2D(proj_data, geo, angles, gpuids):
    return algs.fdk(proj_data, geo, angles, gpuids=gpuids)
   
def reconstruct_chunked(chunk_generator, geo, gpuids):
    reconstructed = None
    for i, (proj_chunk, angles_chunk) in enumerate(chunk_generator):
        print(f"Chunk {i+1}: Reconstructing {proj_chunk.shape}")
        start = time.time()
        img_chunk = algs.fdk(proj_chunk, geo, angles_chunk, gpuids=gpuids)
        #print(proj_chunk.shape)
        #print(angles_chunk.shape)
        print(angles_chunk[:5])
        print(angles_chunk[-5:])
        #(geo)
        if reconstructed is None:
            reconstructed = np.zeros_like(img_chunk, dtype=np.float32)
            #plt.figure()
            #plt.title("Chunk")
            #plt.imshow( img_chunk[1400,:,:], cmap='gray')
            #plt.axis('off')
            #.show()
            reconstructed += img_chunk
            print(img_chunk.shape)
            print(reconstructed.shape)
        else:
            print(f"Before chunk {i+1}, recon sum: {reconstructed.sum():.2f}")
            #plt.figure()
            #plt.title("Chunk")
            #plt.imshow( img_chunk[1400,:,:], cmap='gray')
            #plt.axis('off')
            #plt.show()
            reconstructed += img_chunk
            print(f"After  chunk {i+1}, recon sum: {reconstructed.sum():.2f}")
        #plt.figure()
        #plt.title("Sum")
        #plt.imshow( reconstructed[1400,:,:], cmap='gray')
        #plt.axis('off')
        #plt.show()
        del proj_chunk, img_chunk
        gc.collect()
        time.sleep(3)
        gc.collect()
        time.sleep(3)
        print(f"Chunk {i+1} done in {time.time() - start:.1f} sec")
    return reconstructed