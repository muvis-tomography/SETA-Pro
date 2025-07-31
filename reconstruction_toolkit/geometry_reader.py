
# geometry_reader.py
# -----------------------------------------------------
# Unified geometry reader for Diondo (XML) and Nikon (XtekCT) systems.
#
# Provides:
# - DiondoGeometryReader(xml_file)     → For DIONDO scans
# - NikonGeometryReader(folder_path)   → For NIKON scans
#
# Kullanıcı, GUI üzerinden hangi sistemi seçerse ilgili sınıf kullanılır.
#
# Example usage:
#   reader = DiondoGeometryReader("scan.xml")
#   geo3D = reader.get_geometry("3D")
#
# Author: Kubra Kumrular (RKK)
# -----------------------------------------------------

##% import the libraries 
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from configparser import ConfigParser
import tigre
from tigre.utilities import gpu
import copy

#=========================================
#%  Define the Diondo Geometry from the XML file 

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



#%  Define the Nikon Geometry from the XtekCT file 
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

