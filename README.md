
SETA Recon Pro v1.0 – A Standalone Executable Tool for Automated Image Reconstruction in Cone Beam Computed Tomography


SETA Recon Pro is a user-friendly, standalone executable tool developed to perform automated 3D image reconstruction from Cone Beam CT (CBCT) data. The tool integrates key libraries such as TIGRE and Algotom, providing an accessible solution for users who may not be familiar with complex reconstruction workflows.

The tool is specifically designed to support real-world CBCT data collected from:

📌 DIONDO d5

📌 Nikon XT H 450 / 225 systems

These systems have been tested and validated at the University of Southampton, part of the [National X-ray Computed Tomography facility (NXCT)](https://www.southampton.ac.uk/muvis/about/equipment/d5.page).

🔹 User Interface: A clean and intuitive interface for quick setup and reconstruction.

🔹 Three Reconstruction Modes:

Mode 1: Central Slice Reconstruction
Reconstructs only the central slice of the scanned volume.

Mode 2: Full Volume Reconstruction
Uses an intelligent chunking strategy based on available system memory (10% RAM used for projection upload).

Mode 3: Z-range Reconstruction
Reconstructs a user-defined range of slices using a method described in the accompanying publication.
[Related publication on Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8O9RNFEAAAAJ&sortby=pubdate&citation_for_view=8O9RNFEAAAAJ%3Ap2g8aNsByqUC)

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
OS: Windows (Tested)

GPU: CUDA-compatible GPU (recommended for TIGRE acceleration)

RAM: Min. 8GB (tool auto-manages based on available memory)
