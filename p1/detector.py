"""
detector.py — Détection du marqueur coloré dans UNE image caméra.

Pipeline (chaque étape est une fonction de image_ops.py) :

    image BGR (640x480)
      │ 1. réduction de résolution (/2)        -> moins de calcul, moins de bruit
      │ 2. flou moyenneur 3x3                  -> lisse le bruit du capteur
      │ 3. conversion BGR -> HSV               -> séparer couleur et luminosité
      │ 4. seuillage HSV                       -> masque binaire du marqueur
      │ 5. ouverture + fermeture               -> enlève points isolés, bouche trous
      │ 6. composantes connexes                -> séparer les taches
      │ 7. sélection de la plus grande tache   -> le marqueur (filtre d'aire min/max)
      │ 8. centroïde                           -> position (x, y)
      ▼
    Detection(found, x, y, area, bbox) en pixels de l'image PLEINE résolution

Le détecteur ne lisse rien dans le temps : c'est le rôle de stabilizer.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import image_ops as ops

try:
    import cv2
except ImportError:          # OpenCV n'est obligatoire que pour use_opencv=True
    cv2 = None


@dataclass
class Detection:
    """Résultat brut de la détection sur une image."""
    found: bool
    x: float = float("nan")          # centroïde, pixels pleine résolution
    y: float = float("nan")
    area: int = 0                    # aire en pixels pleine résolution
    bbox: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1 (pleine résolution)
    n_blobs: int = 0                 # nombre de taches candidates (diagnostic)
    reason: str = ""                 # pourquoi rien n'a été trouvé (diagnostic)


class ColorMarkerDetector:
    """Détecteur de marqueur coloré par segmentation HSV."""

    def __init__(self, hsv_lower=(40, 80, 60), hsv_upper=(85, 255, 255),
                 downscale=2, blur=3, open_kernel=3, close_kernel=5,
                 min_area=40, max_area_ratio=0.15, use_opencv=False, **_ignored):
        self.hsv_lower = tuple(int(v) for v in hsv_lower)
        self.hsv_upper = tuple(int(v) for v in hsv_upper)
        self.downscale = max(1, int(downscale))
        self.blur = int(blur)
        self.open_kernel = int(open_kernel)
        self.close_kernel = int(close_kernel)
        self.min_area = int(min_area)                # pixels pleine résolution
        self.max_area_ratio = float(max_area_ratio)  # fraction max de l'image
        self.use_opencv = bool(use_opencv) and cv2 is not None
        self.last_mask = None                        # masque nettoyé (pour affichage)

    # ------------------------------------------------------------------
    def set_thresholds(self, lower, upper):
        self.hsv_lower = tuple(int(v) for v in lower)
        self.hsv_upper = tuple(int(v) for v in upper)

    # ------------------------------------------------------------------
    def compute_mask(self, frame_bgr):
        """Étapes 1 à 5 : renvoie le masque binaire nettoyé (résolution réduite)."""
        f = self.downscale
        if self.use_opencv:
            # Variante OpenCV (mêmes étapes, plus rapide) — utile pour comparer.
            small = cv2.resize(frame_bgr, None, fx=1 / f, fy=1 / f,
                               interpolation=cv2.INTER_AREA) if f > 1 else frame_bgr
            if self.blur > 1:
                small = cv2.blur(small, (self.blur, self.blur))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = ops.in_range_hsv(hsv, self.hsv_lower, self.hsv_upper)
            m = mask.astype(np.uint8)
            if self.open_kernel > 1:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.open_kernel,) * 2, np.uint8))
            if self.close_kernel > 1:
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((self.close_kernel,) * 2, np.uint8))
            return m.astype(bool)

        small = ops.downscale(frame_bgr, f)
        small = ops.box_blur(small, self.blur)
        hsv = ops.bgr_to_hsv(small)
        mask = ops.in_range_hsv(hsv, self.hsv_lower, self.hsv_upper)
        mask = ops.opening(mask, self.open_kernel)
        mask = ops.closing(mask, self.close_kernel)
        return mask

    # ------------------------------------------------------------------
    def detect(self, frame_bgr) -> Detection:
        """Détecte le marqueur dans une image BGR. Renvoie une Detection."""
        f = self.downscale
        mask = self.compute_mask(frame_bgr)
        self.last_mask = mask

        n_white = int(mask.sum())
        total = mask.size
        if n_white == 0:
            return Detection(False, reason="aucun pixel de la couleur")
        # Garde-fou : si une grosse partie de l'image passe le seuil, les
        # seuils sont mauvais (ou la couleur est projetée sur le tableau !).
        if n_white > 0.5 * total:
            return Detection(False, reason="masque trop grand : seuils HSV a revoir")

        if self.use_opencv:
            n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            n -= 1
        else:
            labels, n = ops.connected_components(mask)
        blobs = ops.component_stats(labels, n)

        min_a = self.min_area / (f * f)              # aire min en pixels réduits
        max_a = self.max_area_ratio * total
        candidates = [b for b in blobs if min_a <= b["area"] <= max_a]
        if not candidates:
            return Detection(False, n_blobs=len(blobs),
                             reason="taches trop petites ou trop grandes")

        best = candidates[0]                         # la plus grande tache valide

        # Retour en coordonnées pleine résolution.
        # Le pixel réduit i couvre les pixels [i*f, i*f+f-1] -> centre = i*f + (f-1)/2
        x = best["cx"] * f + (f - 1) / 2.0
        y = best["cy"] * f + (f - 1) / 2.0
        x0, y0, x1, y1 = best["bbox"]
        bbox = (x0 * f, y0 * f, x1 * f + f - 1, y1 * f + f - 1)
        return Detection(True, x=float(x), y=float(y), area=int(best["area"] * f * f),
                         bbox=bbox, n_blobs=len(candidates))


def sample_hsv_range(frame_bgr, x, y, radius=4, h_margin=10, s_margin=60, v_margin=70):
    """Calcule automatiquement une plage HSV à partir d'un clic sur le marqueur.

    On prend un petit carré autour du clic, on calcule la médiane HSV et on
    ajoute des marges. Utilisé par tools/tune_hsv.py.
    """
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = ops.bgr_to_hsv(frame_bgr[y0:y1, x0:x1]).reshape(-1, 3).astype(int)
    hm, sm, vm = np.median(patch, axis=0).astype(int)
    lower = ((hm - h_margin) % 180, max(0, sm - s_margin), max(0, vm - v_margin))
    upper = ((hm + h_margin) % 180, 255, 255)
    return lower, upper, (int(hm), int(sm), int(vm))
