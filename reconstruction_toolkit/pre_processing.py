
# pre_processing.py
# ------------------------------------------------------------
# Basic pre-processing steps for sinogram data before reconstruction.
#
# Currently supports:
# - Stripe removal using Algotom
# - Zinger removal (spike noise)
# - Padding on detector sides (5% or 10%)
# - Log transform with clipping
#
# Main Function:
# - process_projection(proj, cleaning_mode, padding_mode)
#
# Author: Kubra Kumrular (RKK)
# ------------------------------------------------------------

##% Import the libraries 
import numpy as np
import time
from algotom.prep.correction import flat_field_correction

def process_projection(proj, cleaning_mode=0, padding_mode=0):
    # Apply cleaning filters
    cleaning_applied = False
    padding_applied = False
    start = time.time()

    if cleaning_mode in [2, 3]:
        cleaning_applied = True
        proj = flat_field_correction(
            proj,
            flat=np.ones_like(proj),
            dark=np.zeros_like(proj),
            use_dark=False,
            options={
                "method": "remove_stripe_based_sorting",
                "para1": 15,
                "para2": 1
            }
        )


    if cleaning_mode in [1, 3]:
        cleaning_applied = True
        proj = flat_field_correction(
            proj,
            flat=np.ones_like(proj),
            dark=np.zeros_like(proj),
            use_dark=False,
            options={
                "method": "remove_zinger",
                "para1": 0.002,
                "para2": 1
            }
        )

    # Convert to log
    #raw_proj = np.clip(raw_proj, 1e-6, None)
    proj = np.clip(proj, 1e-6, None)
    proj = -np.log(proj / 60000.0).astype(np.float32)

    # Apply padding
    pad_width = 0
    if padding_mode == 1:
        pad_width = int(proj.shape[2] * 0.05)
        padding_applied = True
    elif padding_mode == 2:
        pad_width = int(proj.shape[2] * 0.10)
        padding_applied = True

    if pad_width > 0:
        if proj.ndim == 3:
            proj = np.pad(proj, ((0, 0), (0, 0), (pad_width, pad_width)), mode='edge')
        elif proj.ndim == 2:
            proj = np.pad(proj, ((0, 0), (pad_width, pad_width)), mode='edge')

    duration = time.time() - start

    #write_log(f"Cleaning + Padding took {duration:.2f} sec")
    cleaning_map = {0: "None", 1: "Zinger", 2: "Stripe", 3: "Zinger + Stripe"}
    padding_map = {0: "None", 1: "5%", 2: "10%"}

    #log_summary = f"[Cleaning: {cleaning_map.get(cleaning_mode)} | Padding: {padding_map.get(padding_mode)}]"

    #if cleaning_applied or padding_applied:
    #    write_log(f"{log_summary} → Sinogram processing took {duration:.2f} sec")
   # else:
       # write_log(f"{log_summary} → Sinogram processing: Skipped")

    cleaning_label = cleaning_map.get(cleaning_mode, "Unknown")
    padding_label = padding_map.get(padding_mode, "Unknown")

    return proj, pad_width, cleaning_applied, padding_applied, cleaning_label, padding_label, duration


