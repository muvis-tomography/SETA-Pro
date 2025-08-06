
SETA Recon Pro v1.0 – A Standalone Executable Tool for Automated Image Reconstruction in Cone Beam Computed Tomography

SETA Recon Pro is a user-friendly, standalone executable tool developed to perform automated 3D image reconstruction from Cone Beam CT (CBCT) data. The tool integrates key libraries such as TIGRE and Algotom, providing an accessible solution for users who may not be familiar with complex reconstruction workflows.

## 🔽 Download

You can download the latest standalone executable here:  
[**Download SETA Recon Pro v1.0**](https://github.com/muvis-tomography/SETA-Lite/releases/tag/v1.0)

> No installation required. Just run the `.exe` file.

The tool is specifically designed to support real-world CBCT data collected from:

📌 [DIONDO d5](https://muvis.org/equipment/nxct-diondo-d5)  
📌 [Nikon XT H 450 / 225 systems](https://muvis.org/equipment/custom-450-225-kvp-hutch)

These systems have been tested and validated at the [μ-VIS X-Ray Imaging Centre](https://muvis.org/), University of Southampton, as part of the [National X-ray Computed Tomography facility (NXCT)](https://muvis.org/).

🔹 User Interface: A clean and intuitive interface for quick setup and reconstruction.

🔹 Three Reconstruction Modes:

Mode 1: Central Slice Reconstruction
Reconstructs only the central slice of the scanned volume.

Mode 2: Full Volume Reconstruction
Uses an intelligent chunking strategy based on available system memory (10% RAM used for projection upload).

Mode 3: Z-range Reconstruction  
Reconstructs a user-defined range of slices using a method described in the accompanying publication.  

**Scalable FBP decomposition for cone-beam CT reconstruction**  
_P. Chen, M. Wahib, X. Wang, T. Hirofuchi, H. Ogawa, A. Biguri, R. Boardman, T. Blumensath, S. Matsuoka_  
[🔗 View on Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Scalable+FBP+decomposition+for+cone-beam+CT+reconstruction&btnG=)  
[📄 DOI (acm.org)](https://doi.org/10.1145/3458817.3476139)  

🔹 Automatic Center of Rotation Estimation
This ensures accurate alignment before reconstruction.

🔹 Processing:
This step applies a set of enhancements to improve reconstruction quality. It includes:

Padding before reconstruction to handle truncated projections caused by limited field of view, mitigating ring artifacts from cone-beam effects.

Masking after reconstruction to suppress noise outside the region of interest.

Sinogram filtering and cleaning, implemented via Algotom, to reduce streaks and noise prior to reconstruction.

🔹 Compatibility with other machines (e.g., Nikon and Diondo):
Thanks to a modular reader design, support for new devices can be added easily.

🧱 Architecture
TIGRE – GPU-accelerated iterative and analytical reconstruction algorithms.

Algotom – Advanced sinogram preprocessing in sinogram domain

🖥️ System Requirements
OS: Windows 

GPU: CUDA-compatible GPU (recommended for TIGRE acceleration)

RAM: Min. 8GB (tool auto-manages based on available memory)

**Tested configuration:**
The executable version was tested on the following system:

- OS: Windows Server 2022  
- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)  
- CUDA Toolkit: 11.8  
- Python: 3.10 
- RAM: 256 GB  

Key packages:

- pytigre==3.0.0  
- algotom==1.6.0  
- numpy, scipy, numba, matplotlib  

This configuration supports all three reconstruction modes with GPU acceleration via TIGRE and sufficient memory for large-volume processing.
