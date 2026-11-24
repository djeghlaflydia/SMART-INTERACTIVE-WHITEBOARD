"""
tracker.py — POINT D'ENTRÉE DE P1 pour le reste de l'équipe.

PointerTracker = Camera + ColorMarkerDetector + PointerStabilizer,
exécutés dans un THREAD séparé : l'interface (P5) n'est jamais bloquée par
la caméra, et P4 peut lire la dernière position à tout moment.

Utilisation (P4 / programme principal) :

    from p1 import PointerTracker

    tracker = PointerTracker("config.json")
    tracker.start()
    ...
    state = tracker.get_pointer()          # PointerState
    pos = state.position()                 # (x, y) en pixels caméra, ou None
    if pos is not None:
        board_xy = p2_mapper.image_to_board(*pos)   # <- module de P2
    ...
    tracker.stop()

Utilisation (P2, calibration) :

    frame = tracker.get_frame()            # dernière image caméra (copie)
    # -> P2 clique / détecte les 4 coins du tableau sur cette image
"""

import json
import os
import threading
import time

from .camera import Camera
from .detector import ColorMarkerDetector
from .stabilizer import PointerStabilizer, PointerState

DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 30,
               "backend": "auto", "autofocus": False, "auto_exposure": True},
    "detection": {"method": "color_marker",
                  "hsv_lower": [40, 80, 60], "hsv_upper": [85, 255, 255],
                  "downscale": 2, "blur": 3, "open_kernel": 3, "close_kernel": 5,
                  "min_area": 40, "max_area_ratio": 0.15, "use_opencv": False},
    "stabilization": {"alpha_min": 0.25, "alpha_max": 0.9, "speed_scale": 25.0,
                      "dead_zone": 1.5, "max_lost_frames": 5, "max_jump": 150.0},
}


def load_config(path="config.json"):
    """Charge config.json et complète avec les valeurs par défaut."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))     # copie profonde
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for section, values in user.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    return cfg


def save_config(cfg, path="config.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class PointerTracker:
    def __init__(self, config_path="config.json", config=None):
        self.config_path = config_path
        self.cfg = config if config is not None else load_config(config_path)
        self.camera = Camera(**self.cfg["camera"])
        self.detector = ColorMarkerDetector(**self.cfg["detection"])
        self.stabilizer = PointerStabilizer(**self.cfg["stabilization"])

        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._frame = None
        self._mask = None
        self._state = PointerState()
        self.frame_index = 0
        self.fps = 0.0
        self.error = None

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Traite UNE image (sans thread). Utile pour les tests ou un usage synchrone."""
        self.frame_index += 1
        det = self.detector.detect(frame)
        h, w = frame.shape[:2]
        return self.stabilizer.update(det, self.frame_index, (w, h))

    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return self
        self.camera.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="P1-tracker", daemon=True)
        self._thread.start()
        # attend la première image (max 3 s) pour que get_frame() ne renvoie pas None
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 3 and self.error is None:
            time.sleep(0.01)
        return self

    def _loop(self):
        t_prev = time.time()
        fails = 0
        while self._running:
            t_start = time.time()
            ok, frame = self.camera.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:
                    self.error = "La caméra ne renvoie plus d'images."
                    with self._lock:
                        self._state = PointerState(frame_index=self.frame_index)
                time.sleep(0.01)
                continue
            fails = 0
            state = self.process_frame(frame)
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            with self._lock:
                self._frame = frame
                self._mask = self.detector.last_mask
                self._state = state
            if self.camera.is_file:                    # vidéo : vitesse réelle
                time.sleep(max(0.0, 1.0 / max(1, self.camera.fps) - (time.time() - t_start)))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.camera.release()

    # ------------------------------------------------------------------
    # API publique (thread-safe)
    def get_pointer(self) -> PointerState:
        """Dernier état du pointeur (pixels caméra)."""
        with self._lock:
            return self._state

    def get_frame(self):
        """Copie de la dernière image caméra (BGR), ou None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_debug(self):
        """(image, masque, état) pour l'affichage de contrôle."""
        with self._lock:
            f = None if self._frame is None else self._frame.copy()
            m = None if self._mask is None else self._mask.copy()
            return f, m, self._state

    @property
    def frame_size(self):
        return self.camera.size

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
