# load_projections.py
# ------------------------------------------------------------
# Projection loading module for CT data.
#
# Supports:
# - Single central slice loading for preview or geometry check (mode=1)
# - Full projection loading in chunks for memory-efficient reconstruction (mode=2)
# - Volume-restricted projection loading with detector V-range optimization (mode=3)
#
# Includes automatic RAM-based chunking, preprocessing, and optional geometry slicing.
#
# Author: Kubra Kumrular (RKK)
# ------------------------------------------------------------



##% import libraries 
import os
import glob
import time
import numpy as np
import psutil
from PIL import Image
from joblib import cpu_count
from multiprocessing import Pool

from .logger import write_log
from .pre_processing import process_projection
from .geometry_utils import compute_ab_vrange_corrected, compute_subgeo_for_zrange
from .geometry_reader import DiondoGeometryReader, NikonGeometryReader
from .user_interface import get_user_inputs

def auto_chunk_size(proj_shape, reader, bytes_per_voxel=4):
       
    """
    Calculates the optimal chunk size for loading projection data based on system RAM.

    Parameters:
        reader: The geometry reader object with get_raw_dimensions() and get_projection_count().
        bytes_per_voxel: Size of each voxel in bytes. Default is 4 for float32.
        ram_fraction: Maximum fraction of system RAM to use (e.g., 0.25 means 25%).

    Returns:
        chunk_size: Number of projections per chunk.
        chunk_count: Total number of chunks.
        """
       
    # Get system RAM
    mem_info = psutil.virtual_memory()
    total_ram_MB = mem_info.total / (1024**2)  # in MB

    # Each proj (N, H, W) float32 
    # Projection info
    total_proj = reader.get_projection_count()
    # Estimate memory for one projection
    #proj_shape = # (H, W)
    single_proj_MB = np.prod(proj_shape) * bytes_per_voxel / (1024**2) 

        # max %10VRAM usage
    safety_factor = 0.1
    max_ram_usage_MB = total_ram_MB * safety_factor 

    safe_proj_per_chunk = max(1, int(max_ram_usage_MB // single_proj_MB))

    # Divide the total_proj count into parts according to safe_proj_per_chunk
    chunk_count = int(np.ceil(total_proj / safe_proj_per_chunk))
    chunk_size = int(np.ceil(total_proj / chunk_count))

    # Does the actual chunk fit into memory?
    used_MB = chunk_size * single_proj_MB
    if used_MB > max_ram_usage_MB:
        chunk_size = safe_proj_per_chunk
        chunk_count = int(np.ceil(total_proj / chunk_size))

    print(f"[auto_chunk_size] {total_proj} proj -> {chunk_count} chunk, {chunk_size} proj each")
    print(f"Estimated memory per chunk: {chunk_size * single_proj_MB:.2f} MB (limit: {max_ram_usage_MB:.2f} MB)")

    return chunk_size
    

#Defining the uploadig function for each mode 

def load_proj_central(f, raw_shape):
    if f.lower().endswith(".tif") or f.lower().endswith(".tiff"):
        img = Image.open(f)
        arr = np.array(img)
        center_row = arr.shape[0] // 2
        return arr[center_row:center_row+1, :]
    else:
        raw = np.fromfile(f, dtype=np.uint16).reshape(raw_shape)
        center_row = raw_shape[0] // 2
        return raw[center_row:center_row+1, :]

def load_proj_chunk(f, raw_shape):
    if f.lower().endswith(".tif") or f.lower().endswith(".tiff"):
        img = Image.open(f)
        arr = np.array(img)
        return arr
    else:
        return np.fromfile(f, dtype=np.uint16).reshape(raw_shape)

def load_proj_zrange(f, raw_shape, a, b):
    if f.lower().endswith(".tif") or f.lower().endswith(".tiff"):
        img = Image.open(f)
        arr = np.array(img)
        return arr[a:b, :]
    else:
        raw = np.fromfile(f, dtype=np.uint16).reshape(raw_shape)
        return raw[a:b, :]



def upload_projections(xml_file, proj_folder, mode, raw_shape, user_inputs, z_start=None, z_end=None, chunk_size=None):
    import glob
    n_threads = cpu_count() *0.9 # 90 percent of cpu 

    #reader = DiondoGeometryReader(xml_file)
    if user_inputs["reader_type"] == 0:
        reader = DiondoGeometryReader(xml_file)
        n_projs = reader.get_projection_count()
        angles = -np.linspace(0, 2*np.pi, n_projs, endpoint=False)
        file_list = sorted(glob.glob(os.path.join(proj_folder, '*.raw')))[:-1]
    else:
        reader = NikonGeometryReader(os.path.dirname(xml_file))
        n_projs = reader.get_projection_count()
        angles = reader.get_angles()
        scan_folder = os.path.dirname(xml_file)
        #file_list = sorted(glob.glob(os.path.join(scan_folder, '*.tif')))[:n_projs]
        #file_list = sorted(glob.glob(glob.escape(os.path.join(scan_folder, '*.tif'))))[:-1]
        search_pattern = os.path.join(glob.escape(scan_folder), '*.tif')
        file_list = sorted(glob.glob(search_pattern))[:-1]

        if not file_list:
            raise FileNotFoundError(f"No TIFF files found in {scan_folder}")

    #file_list = file_list[:n_projs]
    
   
    if mode == 1:
 
        with Pool(int(n_threads)) as pool:
            proj_list = pool.starmap(load_proj_central, [(f, raw_shape) for f in file_list])
        proj_array = np.stack(proj_list)
        proj_array, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration = (
            process_projection(proj_array, user_inputs["cleaning"], user_inputs["padding"]))
        return proj_array, reader.get_geometry("2D"), angles, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration
    
    if mode == 2:
        pad_width_holder = [None] 
        info_holder = [None]
        if chunk_size is None:
            example_proj_shape = (1, *raw_shape)  # örnek: (1, 3008, 3008)
            #chunk_size, _= auto_chunk_size(example_proj_shape, reader, bytes_per_voxel=4)
            chunk_size= auto_chunk_size(example_proj_shape, reader, bytes_per_voxel=4)
            print(chunk_size)
            #print(chunk_count)

            #chunk_size = chunk_size[0]

        def chunk_generator():
            for i in range(0, n_projs, chunk_size):
                files = file_list[i:i+chunk_size]
                angles_chunk = angles[i:i+chunk_size]
                start = time.time()
                with Pool(int(n_threads)) as pool:
                    proj_chunk  = pool.starmap(load_proj_chunk, [(f, raw_shape) for f in files])
                proj_chunk = [p.reshape(raw_shape) for p in proj_chunk]
                duration = time.time() - start
                write_log(f"[Chunk {i//chunk_size+1}] Loaded {len(files)} projections in {duration:.2f} sec")
                proj_chunk = np.stack(proj_chunk)
                proj_chunk, pad_width , cleaning_applied, padding_applied, cleaning_label, padding_label, duration= (
                    process_projection(proj_chunk, user_inputs["cleaning"], user_inputs["padding"]))
                if pad_width_holder[0] is None:
                    pad_width_holder[0] = pad_width  # save in the first 
                if info_holder[0] is None:
                    info_holder[0] = (cleaning_applied, padding_applied, cleaning_label, padding_label, duration)
                yield proj_chunk, angles_chunk
        return chunk_generator(), reader.get_geometry("3D"),  pad_width_holder, info_holder
    
    
    elif mode == 3:
        geo_full = reader.get_geometry("3D")
        a, b = compute_ab_vrange_corrected(geo_full, z_start, z_end)
        v_height = b - a
        with Pool(int(n_threads)) as pool:
            proj_list = pool.starmap(load_proj_zrange, [(f, raw_shape, a, b) for f in file_list])
        proj_array = np.stack(proj_list)
        proj_array, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration = (
            process_projection(proj_array, user_inputs["cleaning"], user_inputs["padding"]))
        geo= compute_subgeo_for_zrange(geo_full, z_start, z_end, a, b)
        #print(proj_array.shape)
        #print(geo)
        return proj_array, geo, angles, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration

    else:
        raise ValueError("Invalid mode.")



# this for saving the sigram

def load_sample_sinogram(reader_type, xml_file, raw_shape=None, projection_count=None, proj_folder=None):
   
    if reader_type == 0:
        # DIONDO - RAW
        if proj_folder is None:
            raise ValueError("proj_folder is required for RAW (Diondo) reader.")
        file_list = sorted(glob.glob(os.path.join(proj_folder, '*.raw')))
        if projection_count:
            file_list = file_list[:projection_count]
        sino = [np.fromfile(f, dtype=np.uint16).reshape(raw_shape)[raw_shape[0] // 2] for f in file_list]
    else:
        # NIKON - TIFF
        folder = os.path.dirname(xml_file)
        file_list = sorted(glob.glob(os.path.join(folder, '*.tif')))
        if projection_count:
            file_list = file_list[:projection_count]
        sino = [np.array(Image.open(f))[np.array(Image.open(f)).shape[0] // 2] for f in file_list]

    sino = np.stack(sino)
    return -np.log(np.clip(sino, 1e-6, None) / 60000.0).astype(np.float32)
