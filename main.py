# main.py
# ------------------------------------------------------------------------------
# Master Script for Automatic CT Reconstruction - Diondo & Nikon Systems
#
# This script orchestrates the full CT reconstruction pipeline using TIGRE:
# - Gets user inputs from GUI
# - Loads projections (full, central, or z-range)
# - Applies preprocessing (cleaning, padding)
# - Estimates or sets COR (Center of Rotation)
# - Builds geometry (TIGRE Geo object)
# - Runs FDK reconstruction (2D/3D)
# - Applies post-processing (e.g. circular masking)
# - Saves central slice, sinogram, and/or full volume
# - Displays final result to user
#
# Dependencies:
#   - TIGRE
#   - Pillow, matplotlib, numpy
#   - Custom modules: logger, geometry_reader, user_interface, etc.
#
# Author: Kubra Kumrular (RKK)


# Library imports
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from itertools import tee, chain
from tkinter import Tk
import multiprocessing

# TIGRE
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu

# Custom modules
from SETA_Pro.reconstruction_toolkit import (
    calculate_COR,
    write_log, log_step,
    DiondoGeometryReader, NikonGeometryReader,
    get_user_inputs,
    upload_projections, auto_chunk_size, load_sample_sinogram,
    main_reconstruction_flow,
    save_outputs,
    apply_circular_mask_inplace_parallel,
    process_projection,
    compute_ab_vrange_corrected, trim_geo_after_padding, compute_subgeo_for_zrange
)
from SETA_Pro.reconstruction_toolkit import logger


gpu_names = gpu.getGpuNames()         # ['NVIDIA GeForce RTX 3060']
gpuids = gpu.getGpuIds(gpu_names[0])  # 


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. Get user inputs
    user_inputs = get_user_inputs()

    # === 2. Get base folder and case name ===
    xml_path = user_inputs["xml_file"]
    base_dir = os.path.dirname(xml_path)  # Example: .../model_engine/
    case_name = os.path.basename(base_dir)  # Example: model_engine

    # === 3. Create output dir and log file dynamically ===
    output_dir = os.path.join(base_dir, f"output_{case_name}")
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_PATH = os.path.join(output_dir, f"recon_log_{case_name}_{now}.txt")
    logger.set_log_path(LOG_FILE_PATH)

    # === 4. Log start info ===
    write_log(f"Reconstruction Started for case: {case_name}")
    write_log(f"Geometry info path: {xml_path}")
    write_log(f"Proj folder: {user_inputs['proj_folder']}")
    write_log(f"Output directory: {output_dir}")

    # === 5. Continue reconstruction pipeline ===
    if user_inputs["reader_type"] == 0:
        geo_reader = DiondoGeometryReader(xml_path)
    else:
        geo_reader =  NikonGeometryReader(os.path.dirname(xml_path))

    raw_shape = geo_reader.get_raw_dimensions()
    mode = user_inputs["mode"]

    # === Log Original Geometry ===
    geo_original = geo_reader.get_geometry("2D" if mode == 1 else "3D")
    write_log("="*60)
    write_log("ORIGINAL GEOMETRY (from file):")
    write_log(str(geo_original))
    write_log("="*60)

    if mode == 2:
    # Mode 2'de gerçekten yükleme chunks içinde yapılıyor
        upload_result = upload_projections(
            xml_file=xml_path,
            proj_folder=user_inputs["proj_folder"],
            mode=mode,
            raw_shape=raw_shape,
            user_inputs=user_inputs, 
            z_start=user_inputs["z_start"],
            z_end=user_inputs["z_end"],
            chunk_size=auto_chunk_size((1, *raw_shape), geo_reader)
        )
    else:
        with log_step("Uploading Projections"):
            upload_result = upload_projections(
                xml_file=xml_path,
                proj_folder=user_inputs["proj_folder"],
                mode=mode,
                raw_shape=raw_shape,
                user_inputs=user_inputs, 
                z_start=user_inputs["z_start"],
                z_end=user_inputs["z_end"],
                chunk_size=None
        )

    if mode == 2:
        chunk_generator_fn, geo, pad_width_holder, info_holder = upload_result
        #temp_gen = chunk_generator_fn
        #first_chunk = next(temp_gen)
        #pad_width = pad_width_holder[0] 
        
        #chunk_gen = chain([first_chunk], chunk_generator_fn)

        #chunk_gen, geo, pad_width_holder, info_holder = upload_result
        # = next(chunk_gen)
        #pad_width = pad_width_holder[0] 
        # rechain(first, rest_gen):
        #    yield first
        #    yield from rest_gen
        #chunk_gen = rechain(first_chunk, chunk_gen)
        chunk_gen1, chunk_gen2 = tee(chunk_generator_fn)
        first_chunk = next(chunk_gen1)
        pad_width = pad_width_holder[0]
        chunk_gen = chain([first_chunk], chunk_gen1)


        proj_data = None
        angles = -np.linspace(0, 2*np.pi, geo_reader.get_projection_count(), endpoint=False)

        if info_holder[0] is not None:
            cleaning_applied, padding_applied, cleaning_label, padding_label, duration = info_holder[0]
        else:
            cleaning_map = {0: "None", 1: "Zinger", 2: "Stripe", 3: "Zinger + Stripe"}
            padding_map = {0: "None", 1: "5%", 2: "10%"}
            cleaning_label = cleaning_map.get(user_inputs["cleaning"], "None")
            padding_label = padding_map.get(user_inputs["padding"], "None")
            duration = 0.0
    else:
        proj_data, geo, angles, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration = upload_result
        chunk_gen = None

    write_log(f"Preprocessing Summary: Cleaning={cleaning_label}, Padding={padding_label}, Duration={duration:.2f} sec")
    if mode == 3:
        write_log(f"Selected Z-Range: z_start={user_inputs['z_start']}, z_end={user_inputs['z_end']}")


    with log_step("Reconstruction"):
        recon, proj_central,cor_value = main_reconstruction_flow(
            mode=mode,
            proj_data=proj_data,
            geo_reader=geo_reader,
            angles=angles,
            chunk_generator=chunk_gen,
            gpuids=gpuids,
            geo=geo,
            z_start=user_inputs["z_start"],
            z_end=user_inputs["z_end"],
            cor_mode=user_inputs["cor_mode"],
            cor_value=user_inputs["cor_value"],
            apply_mask=bool(user_inputs["masking"]),
            output_dir=output_dir, # ,
            pad_width=pad_width, #upload_result[-1],
            raw_shape=raw_shape,                    # 
            user_inputs=user_inputs 
        )

    with log_step("Saving Output"):
        save_outputs(output_dir, recon, mode, user_inputs, 
                     proj_data=proj_central, cor_value=cor_value, 
                     z_start=user_inputs["z_start"],
                     z_end=user_inputs["z_end"])
    # === Final display ===
    if recon.ndim == 3:
        center_z = recon.shape[0] // 2
        slice_img = recon[center_z, :, :]
    elif recon.ndim == 2:
        slice_img = recon
    elif recon.ndim == 1:
        slice_img = recon[np.newaxis, :]
    else:
        raise ValueError(f"Unexpected shape for recon image: {recon.shape}")
    
    #print(slice_img.shape)

    plt.figure()
    plt.title("Central Slice (Final)")
    plt.imshow(slice_img, cmap='gray')
    plt.axis('off')
    plt.show()

    write_log("Reconstruction finished successfully.")
