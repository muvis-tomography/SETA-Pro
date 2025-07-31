
# --------------------------------------------------------------------------
# This script is for the automatic recon for Diondo 
# Coded by Kubra Kumrular (RKK)
# --------------------------------------------------------------------------
''' 
This script demonstrates how to upload DIOND5 data, perform reconstruction, 
and apply preprocessing and postprocessing steps.
'''
#%Define log file 

import datetime, os, time

LOG_FILE_PATH = None  # Will be set after user_inputs

os.makedirs("output", exist_ok=True)

def log_step(step_name):
    """
    Log a step: prints to terminal and writes to log file.
    Usage:
        with log_step("Loading projections"):
            # do stuff
    """
    class LogContext:
        def __enter__(self):
            self.start_time = time.time()
            self._log(f"{step_name} started...")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            self._log(f"{step_name} completed in {elapsed:.2f} sec")

        def _log(self, message):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {message}"
            print(line)
            with open(LOG_FILE_PATH, "a") as f:
                f.write(line + "\n")

    return LogContext()

def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(line + "\n")

##% Define the libraries 
import tigre
import numpy as np
import tigre.algorithms as algs
from PIL import Image
import matplotlib.pyplot as plt
import os
import gc
import xml.etree.ElementTree as ET
from tigre.utilities import gpu
import time
import xml.etree.ElementTree as ET
import glob
from algotom.prep.removal import remove_stripe_based_sorting
from algotom.prep.correction import flat_field_correction
import copy
from joblib import Parallel, delayed, cpu_count
from tkinter import Tk, Label, Button, IntVar, Radiobutton, Checkbutton, Entry, Frame, filedialog, messagebox,font, simpledialog
import psutil
from scipy.ndimage import sobel, gaussian_filter
#import tifffile as tiff
import multiprocessing
from multiprocessing import Pool, set_start_method
from configparser import ConfigParser
from itertools import tee, chain
import math
set_start_method("spawn", force=True)  # EXE important 

gpu_names = gpu.getGpuNames()         # ['NVIDIA GeForce RTX 3060']
gpuids = gpu.getGpuIds(gpu_names[0])  # 

#%  Define the Geometry from the XML file 

class DiondoGeometryReader:
    """
    Optimized reader for Diondo XML metadata and projection data (RAW format)
    Defined 2D and 3D geometry, depending on teh selsected geometyr mode ( 2D and 3D geo )

    """

    def __init__(self, xml_file):
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        
        self.xml_file = xml_file
        self.geo = None
        self.projection_count = None
        self._parse_xml()

    def _parse_xml(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        geom = root.find("Geometrie")
        recon = root.find("Recon")

        # Required parameters
        DSD = float(geom.find("SourceDetectorDist").text)
        DSO = float(geom.find("SourceObjectDist").text)

        nDetectorX = int(recon.find("ProjectionDimX").text)
        nDetectorY = int(recon.find("ProjectionDimY").text)
        dDetectorX = float(recon.find("ProjectionPixelSizeX").text)
        dDetectorY = float(recon.find("ProjectionPixelSizeY").text)

        voxel_size = float(recon.find("VolumeVoxelSizeX").text)  # assume isotropic
        nVoxelX = int(recon.find("VolumeDimX").text)
        nVoxelY = int(recon.find("VolumeDimY").text)
        nVoxelZ = int(recon.find("VolumeDimZ").text)

        # Projection count
        try:
            self.projection_count = int(recon.find("ProjectionCount").text)
        except:
            raise ValueError("Cannot find <ProjectionCount> in XML")

        # Create TIGRE 3D - geo
        geo3D                   = tigre.geometry()
        geo3D.DSD               = DSD
        geo3D.DSO               = DSO
        geo3D.nDetector         = np.array([nDetectorY, nDetectorX])
        geo3D.dDetector         = np.array([dDetectorY, dDetectorX])
        geo3D.sDetector         = geo3D.nDetector * geo3D.dDetector

        geo3D.nVoxel            = np.array([nVoxelZ, nVoxelY, nVoxelX])
        geo3D.sVoxel            = geo3D.nVoxel * voxel_size
        geo3D.dVoxel            = geo3D.sVoxel / geo3D.nVoxel

        geo3D.offOrigin         = np.array([0, 0, 0])
        geo3D.offDetector       = np.array([0, 0])
        geo3D.rotDetector       = np.array([0, 0, 0])
        geo3D.accuracy          = 0.5
        geo3D.COR               = 0  # y direction displacement for
        geo3D.mode              = "cone"

        self.geo3D              = geo3D
        #print(" 3D - Diondo geometry loaded successfully.")

        # Create TIGRE 2D - geo
        geo2D                   = tigre.geometry()
        geo2D.DSD               = DSD
        geo2D.DSO               = DSO

        geo2D.nDetector         = np.array([1, nDetectorX])
        geo2D.dDetector         = np.array([1.0, dDetectorX])
        geo2D.sDetector         = geo2D.nDetector * geo2D.dDetector

        geo2D.nVoxel            = np.array([1.0, nVoxelY, nVoxelX ])
        geo2D.sVoxel            = geo2D.nVoxel * voxel_size
        geo2D.dVoxel            = geo2D.sVoxel / geo2D.nVoxel

        geo2D.offOrigin         = np.array([0, 0, 0])
        geo2D.offDetector       = np.array([0, 0])
        geo2D.rotDetector       = np.array([0, 0, 0])
        geo2D.accuracy          = 0.5  # Variable to define accuracy of
        geo2D.COR               = 0  # y direction displacement for
        geo2D.mode              = "cone"

        self.geo2D              = geo2D
        #print(" 2D - Diondo geometry loaded successfully.")


    def get_raw_dimensions(self):
        """Returns raw projection dimensions as (height, width) for loading RAW files."""
        recon_tree = ET.parse(self.xml_file).getroot().find("Recon")
        height = int(recon_tree.find("ProjectionDimY").text)
        width = int(recon_tree.find("ProjectionDimX").text)
        return (height, width)

    def get_geometry(self, mode="3D"):
        if mode == "2D":
            return self.geo2D
        elif mode == "3D":
            return self.geo3D
        else:
            raise ValueError(f"Unknown geometry mode: {mode}")

    def get_projection_count(self):
        return self.projection_count




class NikonGeometryReader:
    def __init__(self, folder_path):
        self.folder = folder_path
        self.geo3D = None
        self.geo2D = None
        self.angles = None
        self.raw_shape = None
        self.projection_count = None
        self._parse_geometry()

    def _parse_geometry(self):
        # Find .xtekct file
        files = [f for f in os.listdir(self.folder) if f.endswith(".xtekct")]
        if not files:
            raise FileNotFoundError("No .xtekct file found in folder.")
        #ini_path = os.path.join(self.folder, files[0])
        ini_path = glob.glob(glob.escape(os.path.join(self.folder, files[0])))

        cfg = ConfigParser()
        cfg.read(ini_path)
        cfg = cfg["XTekCT"]

        # Parse values from config
        voxel_size = float(cfg["VoxelSizeX"])  # assuming isotropic
        nVoxelX = int(cfg["RegionPixelsX"])
        nVoxelY = int(cfg["RegionPixelsY"])
        nVoxelZ = int(cfg["RegionPixelsX"])  # assuming isotropic

        dDetectorX = float(cfg["DetectorPixelSizeX"])
        dDetectorY = float(cfg["DetectorPixelSizeY"])
        nDetectorX = int(cfg["DetectorPixelsX"])
        nDetectorY = int(cfg["DetectorPixelsY"])

        DSO = float(cfg["SrcToObject"])
        DSD = float(cfg["SrcToDetector"])

        #COR = -float(cfg.get("CentreOfRotationTop", 0.0))

        self.projection_count = int(cfg["Projections"])

        # ANGLES
        try:
            step = float(cfg["AngularStep"])
            init_angle = float(cfg["InitialAngle"])
            self.angles = -np.deg2rad(np.arange(self.projection_count) * step + init_angle)
        except:
            raise ValueError("Cannot parse angles from .xtekct file.")

        # RAW IMAGE SHAPE
        self.raw_shape = (nDetectorY, nDetectorX)

        # -- GEOMETRY 3D --
        geo3D = tigre.geometry()
        geo3D.accuracy = 0.5
        geo3D.DSO = DSO
        geo3D.DSD = DSD
        geo3D.nVoxel = np.array([nVoxelZ, nVoxelY, nVoxelX])
        geo3D.sVoxel = geo3D.nVoxel * voxel_size
        geo3D.dVoxel = geo3D.sVoxel / geo3D.nVoxel
        geo3D.offOrigin = np.array([0, 0, 0])
        geo3D.nDetector = np.array([nDetectorY, nDetectorX])
        geo3D.dDetector = np.array([dDetectorY, dDetectorX])
        geo3D.sDetector = geo3D.nDetector * geo3D.dDetector
        geo3D.offDetector = np.array([0, 0])
        geo3D.rotDetector = np.array([0, 0, 0])
        #geo3D.COR = COR
        geo3D.mode = "cone"
        self.geo3D = geo3D

        # -- GEOMETRY 2D --
        geo2D = tigre.geometry()
        geo2D.DSO = DSO
        geo2D.DSD = DSD
        geo2D.accuracy = 0.5
        geo2D.nVoxel = np.array([1, nVoxelY, nVoxelX])
        geo2D.dVoxel = np.array([voxel_size]*3)
        geo2D.sVoxel = geo2D.nVoxel * geo2D.dVoxel
        geo2D.offOrigin = np.array([0, 0, 0])
        geo2D.nDetector = np.array([1, nDetectorX])
        geo2D.dDetector = np.array([1.0, dDetectorX])
        geo2D.sDetector = geo2D.nDetector * geo2D.dDetector
        geo2D.offDetector = np.array([0, 0])
        geo2D.rotDetector = np.array([0, 0, 0])
        geo2D.mode = "cone"
        self.geo2D = geo2D

    def get_geometry(self, mode="3D"):
        if mode == "2D":
            return self.geo2D
        elif mode == "3D":
            return self.geo3D
        else:
            raise ValueError("Mode must be '2D' or '3D'.")

    def get_angles(self):
        return self.angles

    def get_raw_dimensions(self):
        return self.raw_shape

    def get_projection_count(self):
        return self.projection_count


#% user inputrs 

def get_user_inputs():
    def submit():
        try:
            inputs["reader_type"] = reader_var.get()
            inputs["xml_file"] = xml_file
            if reader_var.get() == 0:
                inputs["proj_folder"] = proj_folder  # 
            else:
                inputs["proj_folder"] = os.path.dirname(xml_file)  # 
            #inputs["proj_folder"] = proj_folder
            inputs["mode"] = mode_var.get()
            inputs["cor_mode"] = cor_var.get()
            inputs["cor_value"] = float(cor_entry.get()) if cor_var.get() == 1 else None
            inputs["z_start"] = int(z_start_entry.get()) if mode_var.get() == 3 else None
            inputs["z_end"] = int(z_end_entry.get()) if mode_var.get() == 3 else None
            inputs["padding"] = padding_var.get()
            inputs["masking"] = masking_var.get()
            inputs["cleaning"] = cleaning_var.get()
            inputs["save_sinogram"] = save_sinogram_var.get()
            inputs["save_central_slice"] = save_central_slice_var.get()
            inputs["save_raw"] = save_raw_var.get()
            root.quit()
            root.destroy()  # close GUI'yi 
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    root = Tk()
    header_font = ("Arial", 18, "bold")
    section_font = ("Arial", 14)

    root.title("Reconstruction GUI")

    user_closed_gui = {"value": False} 

    def on_close():
        if messagebox.askyesno("Exit Confirmation", "You have unsaved selections. Are you sure you want to exit?"):
            print("User closed the GUI window. Exiting program.")
            write_log("User closed the GUI window. Program terminated.")
            user_closed_gui["value"] = True
            root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)

    inputs = {}
    xml_file = ""
    proj_folder = ""

    def choose_xml():
        nonlocal xml_file
        if reader_var.get() == 0:
            xml_file = filedialog.askopenfilename(
                title="Select XML File", 
                filetypes=[("XML files", "*.xml")])
        else:
            xml_file = filedialog.askopenfilename(
                title="Select XtekCT File",
                filetypes=[("XTekCT files", "*.xtekct")] )
    
        xml_label.config(text=xml_file)

    def choose_proj():
        nonlocal proj_folder
        proj_folder = filedialog.askdirectory(title="Select Projection Folder")
        proj_label.config(text=proj_folder)

    Label(root, text="0. Select Reader Type", font=header_font).pack(anchor='w')
    reader_var = IntVar(value=0)
    Radiobutton(root, text="Diondo (XML)", variable=reader_var, value=0, font=section_font).pack(anchor='w')
    Radiobutton(root, text="Nikon (XtekCT)", variable=reader_var, value=1, font=section_font).pack(anchor='w')

    Label(root, text="1. Select Geometry Info File", font=header_font).pack(anchor='w')
    Button(root, text="Browse", command=choose_xml, font=section_font).pack(anchor='w')
    xml_label = Label(root, text="No file selected")
    xml_label.pack(anchor='w')

    Label(root, text="2. Select Projection Folder", font=header_font).pack(anchor='w')
    Button(root, text="Browse Folder", command=choose_proj, font=section_font).pack(anchor='w')
    proj_label = Label(root, text="No folder selected")
    proj_label.pack(anchor='w')

    Label(root, text="3. Reconstruction Mode", font=header_font).pack(anchor='w')
    mode_var = IntVar(value=1)
    for text, val in [("1 = Central Slice (2D)", 1), ("2 = Full Volume (3D)", 2), ("3 = Z-range (3D)", 3)]:
        Radiobutton(root, text=text, variable=mode_var, value=val, font=section_font).pack(anchor='w')

    z_frame = Frame(root)
    Label(z_frame, text="Z Start:").pack(side='left')
    z_start_entry = Entry(z_frame, width=5, font=section_font)
    z_start_entry.pack(side='left')
    Label(z_frame, text="Z End:").pack(side='left')
    z_end_entry = Entry(z_frame, width=5, font=section_font)
    z_end_entry.pack(side='left')
    z_frame.pack(anchor='w')

    def toggle_z():
        if mode_var.get() == 3:
            z_frame.pack(anchor='w')
        else:
            z_frame.pack_forget()
    mode_var.trace_add("write", lambda *_: toggle_z())

    Label(root, text="4. COR Mode", font=header_font).pack(anchor='w')
    cor_var = IntVar(value=0)
    Radiobutton(root, text="Auto", variable=cor_var, value=0, font=section_font).pack(anchor='w')
    Radiobutton(root, text="Manual", variable=cor_var, value=1, font=section_font).pack(anchor='w')
    Label(root, text="(Manual COR value should be in **millimeters**)", font=("Arial", 10, "italic"), fg="gray").pack(anchor='w')

    cor_entry = Entry(root, font=section_font)
    cor_entry.pack(anchor='w')

    Label(root, text="5. Cleaning", font=header_font).pack(anchor='w')
    cleaning_var = IntVar(value=0)
    for text, val in [("None", 0), ("Zinger", 1), ("Stripe", 2), ("Both", 3)]:
        Radiobutton(root, text=text, variable=cleaning_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="6. Padding", font=header_font).pack(anchor='w')
    padding_var = IntVar(value=0)
    for text, val in [("No Padding", 0), ("5%", 1), ("10%", 2)]:
        Radiobutton(root, text=text, variable=padding_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="7. Masking", font=header_font).pack(anchor='w')
    masking_var = IntVar(value=0)
    for text, val in [("No", 0), ("Yes", 1)]:
        Radiobutton(root, text=text, variable=masking_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="8. Save Options", font=header_font).pack(anchor='w')
    save_sinogram_var = IntVar(value=1)
    save_central_slice_var = IntVar(value=1)
    save_raw_var = IntVar(value=0)
    Checkbutton(root, text="Save Sinogram", variable=save_sinogram_var, font=section_font).pack(anchor='w')
    Checkbutton(root, text="Save Central Slice", variable=save_central_slice_var, font=section_font).pack(anchor='w')
    Checkbutton(root, text="Save RAW Volume", variable=save_raw_var, font=section_font).pack(anchor='w')

    Button(root, text="Submit", command=submit, font=header_font).pack(pady=10)
    root.mainloop()
    #root.destroy()
    if user_closed_gui["value"]:
        os._exit(0)  
    return inputs

def compute_ab_vrange_corrected(geo, z_start, z_end, angles=None):
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

        # max %25VRAM usage
    safety_factor = 0.1
    max_ram_usage_MB = total_ram_MB * safety_factor 

    safe_proj_per_chunk = max(1, int(max_ram_usage_MB // single_proj_MB))

    # Şimdi total_proj sayısını, safe_proj_per_chunk'a göre parçalara bölelim
    chunk_count = int(np.ceil(total_proj / safe_proj_per_chunk))
    chunk_size = int(np.ceil(total_proj / chunk_count))

    # Son güvenlik: Gerçek chunk belleğe sığıyor mu?
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



def upload_projections(xml_file, proj_folder, mode, raw_shape, z_start=None, z_end=None, chunk_size=None):
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
    voxel_size_mm = geo.dVoxel[1] 
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
        geo.rotation_axis_offset = offset
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

    scale = geo.dDetector[1] / geo_binned.dDetector[1]
    coarse_offset_corrected = coarse_offset / scale

    # --- STEP 4: Refined Golden Section Search using full-resolution data ---
    def sharpness_metric(offset):
        geo.COR= offset
        reco = algs.fdk(proj_processed, geo, angles, gpuids=gpuids)
        return np.sum(reco[0] ** 2)

    def golden_section_search(f, a, b, tol=0.001, max_iter=100):
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        for _ in range(max_iter):
            if abs(c - d) < tol:
                break
            if f(c) > f(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
        return (b + a) / 2

    print("Running refined search (GSS)...")
    voxel_mm = geo.dVoxel[1]
    search_a = coarse_offset_corrected  - 2 * voxel_mm
    search_b = coarse_offset_corrected  + 2 * voxel_mm
    refined_offset = golden_section_search(sharpness_metric, search_a, search_b, tol=0.005 * voxel_mm)
    #refined_offset = refined_offset * voxel_mm

    print(f"[COR] Estimated optimal COR offset = {refined_offset:.6f} mm")
    return refined_offset



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


def trim_geo_after_padding(geo, pad_width):
    geo = copy.deepcopy(geo)

    geo.nVoxel[2] -= 2 * pad_width  # X (U axis)
    geo.sVoxel[2] = geo.nVoxel[2] * geo.dVoxel[2]

    geo.nVoxel[1] -= 2 * pad_width  # Y (V axis)
    geo.sVoxel[1] = geo.nVoxel[1] * geo.dVoxel[1]

    geo.nDetector[1] -= 2 * pad_width  # only X axis in detector 
    geo.sDetector[1] = geo.nDetector[1] * geo.dDetector[1]

    return geo




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



def apply_circular_mask_inplace_parallel(recon, geo, radius=0.98):

    """
    Applies a smooth circular mask directly into the existing 3D recon array,
    using parallel processing (RAM dostu).

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




def save_outputs(output_dir, recon, mode, user_inputs, proj_data=None, cor_value=None, z_start=None, z_end=None):

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    case = os.path.basename(output_dir).replace("output_", "")
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


# End block
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


# %%

#hshah
#%%
