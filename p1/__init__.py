"""P1 — Acquisition caméra et détection du pointeur (Smart Interactive Whiteboard)."""

from .camera import Camera
from .detector import ColorMarkerDetector, OpenCVHandDetector, Detection, sample_hsv_range
from .stabilizer import PointerStabilizer, PointerState
from .tracker import PointerTracker, load_config, save_config

__all__ = ["Camera", "ColorMarkerDetector", "OpenCVHandDetector",
           "Detection", "sample_hsv_range",
           "PointerStabilizer", "PointerState", "PointerTracker",
           "load_config", "save_config"]
