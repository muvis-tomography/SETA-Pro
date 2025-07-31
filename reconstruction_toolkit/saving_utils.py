# save_outputs.py
# ------------------------------------------------------------
# Output Saving Module for CT Reconstruction
#
# This module handles the saving of:
# - Central slice (2D PNG)
# - Reconstructed volume (3D RAW format)
# - Sinogram (2D PNG)
#
# It uses user GUI selections (from user_interface.py) and 
# projection geometry (from geometry_reader.py) to determine 
# what and how to save.
#
# Dependencies:
# - logger.py (for logging)
# - projection_loader.py (for reloading sinogram if needed)
#
# Author: Kubra Kumrular (RKK)

import numpy as np
import os 
import matplotlib.pyplot as plt
import datetime 

from .logger import write_log 
from .projection_loader import load_sample_sinogram

def save_outputs(output_dir, recon, mode, user_inputs,
                 proj_data=None, cor_value=None, z_start=None, z_end=None,
                 raw_shape=None, geo_reader=None):
    """
    Saves the outputs of the reconstruction process including central slice,
    full 3D volume (as .raw), and the sinogram.

    Parameters
    ----------
    output_dir : str
        Path where the output files will be saved.

    recon : np.ndarray
        Reconstructed image volume (2D or 3D numpy array).

    mode : int
        Reconstruction mode (1 = 2D central slice, 2 = full 3D, 3 = Z-range 3D).

    user_inputs : dict
        Dictionary of user selections from the GUI, including save options.

    proj_data : np.ndarray, optional
        Raw projection data used for reconstruction (for sinogram saving).

    cor_value : float, optional
        Center of rotation value used in reconstruction.

    z_start : int, optional
        Starting Z index for Z-range reconstructions (mode 3).

    z_end : int, optional
        Ending Z index for Z-range reconstructions (mode 3).

    raw_shape : tuple, optional
        Shape of raw projection images (needed if COR is manually entered and sinogram is requested).

    geo_reader : DiondoGeometryReader or NikonGeometryReader, optional
        Geometry reader object to determine projection count (used if sinogram needs to be reloaded).

    Returns
    -------
    None
        Saves output files to disk and logs the actions.
    """

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    case_name = os.path.basename(output_dir).replace("output_", "")
    shape_str = "x".join(map(str, recon.shape[::-1])) 
    cor_str = f"{cor_value:.3f}".replace(".", "_") if cor_value is not None else "na"

    # === Save central slice ===
    if user_inputs["save_central_slice"]:
        if recon.ndim == 3:
            slice_img = recon[recon.shape[0] // 2]
        elif recon.ndim == 2:
            slice_img = recon
        else:
            slice_img = recon[np.newaxis, :]

        image_name = f"{case_name}_central_slice_{shape_str}.png"
        image_path = os.path.join(output_dir, image_name)
        plt.imsave(image_path, slice_img, cmap='gray')
        write_log(f"Saved CENTRAL SLICE -> {image_path}")

    # === Save raw volume ===
    if user_inputs["save_raw"] and recon.ndim == 3:
        if mode == 3 and z_start is not None and z_end is not None:
            zrange_str = f"z{z_start}to{z_end}_"
        else:
            zrange_str = ""

        volume_name = f"imgFDK_{case_name}_{zrange_str}{shape_str}.raw"

        #volume_name = f"imgFDK_{case_name}_{shape_str}.raw"
        volume_path = os.path.join(output_dir, volume_name)
        recon.astype(np.float32).tofile(volume_path)
        write_log(f"Saved RAW VOLUME -> {volume_path}")

    # === Save sinogram ===
    if user_inputs["save_sinogram"]:
        if proj_data is not None:
            center_row = proj_data.shape[1] // 2 if proj_data.ndim == 3 else 0
            sinogram = proj_data[:, center_row, :] if proj_data.ndim == 3 else proj_data
        else: 
            # if you did not upladed any data (entered COR manually)
            sinogram = load_sample_sinogram(
                reader_type=user_inputs["reader_type"],
                xml_file=user_inputs["xml_file"],
                raw_shape=raw_shape if user_inputs["reader_type"] == 0 else None,
                projection_count=geo_reader.get_projection_count(),
                proj_folder=user_inputs["proj_folder"] if user_inputs["reader_type"] == 0 else None
            )

        sino_shape = f"{sinogram.shape[0]}x{sinogram.shape[1]}"
        sinogram_name = f"sinogram_center_{sino_shape}.png"
        sinogram_path = os.path.join(output_dir, sinogram_name)
        plt.imsave(sinogram_path, sinogram, cmap='gray')
        write_log(f"Saved SINOGRAM -> {sinogram_path}")
