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
    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    frame_size: Tuple[int, int] = (0, 0)   # (largeur, hauteur) de l'image caméra

    def position(self):
        """(x, y) si valide, sinon None. Méthode conseillée pour P2/P4."""
        return (self.x, self.y) if self.valid else None


class PointerStabilizer:
    def __init__(self, alpha_min=0.25, alpha_max=0.9, speed_scale=25.0,
                 dead_zone=1.5, max_lost_frames=5, max_jump=150.0, **_ignored):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.speed_scale = speed_scale
        self.dead_zone = dead_zone
        self.max_lost_frames = max_lost_frames
        self.max_jump = max_jump
        self.reset()

    def reset(self):
        self.x = self.y = None
        self.lost = 10 ** 9          # nb d'images consécutives sans détection
        self._pending = None         # saut en attente de confirmation

    def update(self, det, frame_index=0, frame_size=(0, 0)) -> PointerState:
        st = PointerState(frame_index=frame_index, frame_size=frame_size)

        if det.found:
            st.raw_x, st.raw_y, st.area, st.bbox = det.x, det.y, det.area, det.bbox
            accept = True
            if self.x is None or self.lost > self.max_lost_frames:
                # (ré)apparition : on part directement de la nouvelle position
                self.x, self.y = det.x, det.y
            else:
                d = math.hypot(det.x - self.x, det.y - self.y)
                if d > self.max_jump:
                    # saut suspect : accepté seulement s'il se confirme
                    if self._pending and math.hypot(det.x - self._pending[0],
                                                    det.y - self._pending[1]) < self.max_jump / 3:
                        self.x, self.y = det.x, det.y
                        self._pending = None
                    else:
                        self._pending = (det.x, det.y)
                        accept = False
                else:
                    self._pending = None
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

        if not st.detected:
            if self.x is not None and self.lost <= self.max_lost_frames:
                st.status = "HOLD"
            else:
                st.status = "LOST"

        st.valid = st.status in ("TRACKING", "HOLD")
        if st.valid:
            st.x, st.y = float(self.x), float(self.y)
        return st
