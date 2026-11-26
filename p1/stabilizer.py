"""
stabilizer.py — Stabilisation temporelle de la position détectée.

Problèmes d'une détection brute image par image :
  (a) TREMBLEMENT : même immobile, le centroïde bouge de 1-3 pixels (bruit capteur).
  (b) PERTES BRÈVES : le marqueur disparaît 1-2 images (flou de bouge, main qui passe).
  (c) SAUTS ABERRANTS : une tache parasite de même couleur apparaît une image.

Solutions :
  (a) moyenne mobile exponentielle (EMA) ADAPTATIVE + petite zone morte :
        p_lisse = p_lisse + alpha * (p_brut - p_lisse)
      alpha est petit quand le marqueur bouge peu (on lisse fort -> pas de
      tremblement) et grand quand il bouge vite (on suit -> pas de retard).
  (b) on garde la dernière position pendant max_lost_frames images
      (état « HOLD ») avant de déclarer le pointeur perdu (« LOST »).
  (c) un saut > max_jump pixels n'est accepté que s'il est confirmé sur
      2 images consécutives.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class PointerState:
    """Ce que P1 fournit aux autres modules (P2 / P4).

    Toutes les coordonnées sont en PIXELS DE L'IMAGE CAMÉRA pleine résolution
    (x = colonne, y = ligne, origine en haut à gauche), en float.
    """
    valid: bool = False          # True -> (x, y) est utilisable (détecté OU maintenu)
    detected: bool = False       # True -> marqueur réellement vu sur CETTE image
    status: str = "LOST"         # "TRACKING", "HOLD" (perdu brièvement) ou "LOST"
    x: float = float("nan")      # position lissée (à utiliser)
    y: float = float("nan")
    raw_x: float = float("nan")  # position brute de cette image (diagnostic)
    raw_y: float = float("nan")
    area: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    source: str = "marker"
    hand_bbox: Optional[Tuple[int, int, int, int]] = None
    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    frame_size: Tuple[int, int] = (0, 0)   # (largeur, hauteur) de l'image caméra

    def position(self):
        """(x, y) si valide, sinon None. Méthode conseillée pour P2/P4."""
        return (self.x, self.y) if self.valid else None


class PointerStabilizer:
    def __init__(self, alpha_min=0.25, alpha_max=0.9, speed_scale=25.0,
                 dead_zone=1.5, max_lost_frames=5, max_jump=150.0,
                 filter_type="ema", kalman_q=0.15, kalman_r_still=25.0,
                 kalman_r_move=1.5, **_ignored):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.speed_scale = speed_scale
        self.dead_zone = dead_zone
        self.max_lost_frames = max_lost_frames
        self.max_jump = max_jump
        self.filter_type = filter_type
        self.kalman_q = float(kalman_q)
        self.kalman_r_still = float(kalman_r_still)
        self.kalman_r_move = float(kalman_r_move)
        # Matrices Kalman 2D vitesse constante : état [x, y, vx, vy]^T
        self._F = np.array([[1.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 1.0],
                            [0.0, 0.0, 0.85, 0.0],
                            [0.0, 0.0, 0.0, 0.85]], dtype=np.float64)
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._I4 = np.eye(4, dtype=np.float64)
        self.reset()

    def reset(self):
        self.x = self.y = None
        self.lost = 10 ** 9          # nb d'images consécutives sans détection
        self._pending = None         # saut en attente de confirmation
        self._kf_x = None
        self._kf_P = None

    def _kf_init(self, x, y):
        self._kf_x = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float64)
        self._kf_P = np.diag([4.0, 4.0, 10.0, 10.0])
        self.x, self.y = float(x), float(y)

    def _kf_predict(self):
        Q = np.diag([self.kalman_q * 0.25, self.kalman_q * 0.25,
                     self.kalman_q, self.kalman_q])
        self._kf_x = self._F @ self._kf_x
        self._kf_P = self._F @ self._kf_P @ self._F.T + Q
        self.x, self.y = float(self._kf_x[0]), float(self._kf_x[1])

    def _kf_update(self, mx, my):
        z = np.array([float(mx), float(my)], dtype=np.float64)
        y_innov = z - self._H @ self._kf_x
        d = float(np.hypot(y_innov[0], y_innov[1]))
        # Bruit de mesure R adaptatif : fort lissage à l'arrêt, réactif en mouvement
        t = min(1.0, max(0.0, (d - self.dead_zone) / max(1.0, self.speed_scale)))
        r_val = self.kalman_r_still * (1.0 - t) + self.kalman_r_move * t
        R = np.diag([r_val, r_val])
        S = self._H @ self._kf_P @ self._H.T + R
        K = self._kf_P @ self._H.T @ np.linalg.inv(S)
        self._kf_x = self._kf_x + K @ y_innov
        self._kf_P = (self._I4 - K @ self._H) @ self._kf_P
        self.x, self.y = float(self._kf_x[0]), float(self._kf_x[1])

    def update(self, det, frame_index=0, frame_size=(0, 0)) -> PointerState:
        st = PointerState(frame_index=frame_index, frame_size=frame_size)

        if det.found:
            st.raw_x, st.raw_y, st.area, st.bbox = det.x, det.y, det.area, det.bbox
            st.source = getattr(det, 'source', 'marker')
            st.hand_bbox = getattr(det, 'hand_bbox', None)
            accept = True
            if self.x is None or self.lost > self.max_lost_frames:
                # (ré)apparition : on part directement de la nouvelle position
                if self.filter_type == "kalman":
                    self._kf_init(det.x, det.y)
                else:
                    self.x, self.y = det.x, det.y
            else:
                if self.filter_type == "kalman":
                    self._kf_predict()
                d = math.hypot(det.x - self.x, det.y - self.y)
                if d > self.max_jump:
                    # saut suspect : accepté seulement s'il se confirme
                    if self._pending and math.hypot(det.x - self._pending[0],
                                                    det.y - self._pending[1]) < self.max_jump / 3:
                        if self.filter_type == "kalman":
                            self._kf_init(det.x, det.y)
                        else:
                            self.x, self.y = det.x, det.y
                        self._pending = None
                    else:
                        self._pending = (det.x, det.y)
                        accept = False
                else:
                    self._pending = None
                    if self.filter_type == "kalman":
                        if d > self.dead_zone:
                            self._kf_update(det.x, det.y)
                    else:
                        if d > self.dead_zone:
                            alpha = min(self.alpha_max,
                                        max(self.alpha_min, d / self.speed_scale))
                            self.x += alpha * (det.x - self.x)
                            self.y += alpha * (det.y - self.y)
            if accept:
                self.lost = 0
                st.detected = True
                st.status = "TRACKING"
            else:
                self.lost += 1
        else:
            self.lost += 1
            self._pending = None
            if self.filter_type == "kalman" and self._kf_x is not None and self.lost <= self.max_lost_frames:
                self._kf_predict()

        if not st.detected:
            if self.x is not None and self.lost <= self.max_lost_frames:
                st.status = "HOLD"
            else:
                st.status = "LOST"

        st.valid = st.status in ("TRACKING", "HOLD")
        if st.valid:
            st.x, st.y = float(self.x), float(self.y)
        return st
