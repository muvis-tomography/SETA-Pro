# post_processing.py
# ------------------------------------------------------------
# Post-processing module for reconstruction volume data.
#
# Currently includes:
# - apply_circular_mask_inplace_parallel: Applies a smooth circular mask 
#   to reconstructed 2D or 3D volumes in-place using parallelization.
#
# This helps to suppress edge artifacts after CT reconstruction by masking 
# out regions outside the desired circular region.
#
# Author: Kubra Kumrular (RKK)

##% Define the libraries 
import numpy as np
import copy
from joblib import Parallel, delayed
import math

#========================================

def apply_circular_mask_inplace_parallel(recon, geo, radius=0.98):

    """
    Applies a smooth circular mask directly into the existing 2D-3D recon array,
    using parallel processing.

    Parameters
    ----------
    recon : np.ndarray
        3D volume (nz, ny, nx)
    geo : TIGRE geometry object
    radius : float
        % of the max diameter to retain (default 0.99)
    n_jobs : int
        Number of parallel threads (cores)

    Returns
    -------
    recon : np.ndarray
        Masked 3D volume (modified in-place)
    """
    from joblib import cpu_count
    n_jobs = cpu_count()

    # TIGRE geometry 
    voxel_size_y = geo.dVoxel[1]
    voxel_size_x = geo.dVoxel[2]
    size_y = geo.sVoxel[1]
    size_x = geo.sVoxel[2]

    if recon.ndim == 2:  # 2D slice
        ny, nx = recon.shape
        nz = None
    elif recon.ndim == 3:  # 3D volume
        nz, ny, nx = recon.shape
    else:
        raise ValueError("Unsupported shape for reconstruction array")

    # # Mask creation
    y_range = (ny - 1) / 2
    x_range = (nx - 1) / 2
    Y, X = np.ogrid[-y_range:y_range+1, -x_range:x_range+1]
    dist = np.sqrt((X * voxel_size_x) ** 2 + (Y * voxel_size_y) ** 2)
    r = radius * max(size_x, size_y) / 2
    w = ((voxel_size_x * voxel_size_y) / np.pi) ** 0.5

    mask = (r - dist).clip(-w, w)
    mask *= (0.5 * np.pi) / w
    np.sin(mask, out=mask)
    mask = (0.5 + 0.5 * mask).astype(np.float32)

    # Parallel masking
    if nz is None:  # 2D one slice
        recon *= mask
    else:  # 3D volume ise paralel uygula
        def apply_mask_to_slice(i):
            recon[i] *= mask  # in-place masking

        Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(apply_mask_to_slice)(i) for i in range(nz)
        )

    return recon

