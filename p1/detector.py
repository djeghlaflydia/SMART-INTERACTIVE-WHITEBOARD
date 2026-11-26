"""
detector.py — Détection du pointeur dans UNE image caméra.

Méthodes disponibles :
  1) color_marker   — segmentation HSV d'un marqueur coloré (original)
  2) hand_opencv    — détection de la main par segmentation de la peau (YCrCb + HSV),
                      soustraction de fond MOG2, analyse de contour, enveloppe convexe
                      et localisation du doigt pointeur via OpenCV.

Le détecteur ne lisse rien dans le temps : c'est le rôle de stabilizer.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import image_ops as ops

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class Detection:
    """Résultat brut de la détection sur une image."""
    found: bool
    x: float = float("nan")          # centroïde ou bout de doigt, pixels pleine résolution
    y: float = float("nan")
    area: int = 0                    # aire en pixels pleine résolution
    bbox: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1 (pleine résolution)
    n_blobs: int = 0                 # nombre de taches candidates (diagnostic)
    reason: str = ""                 # pourquoi rien n'a été trouvé (diagnostic)
    source: str = "marker"           # "marker", "hand", "hand_fallback"
    hand_bbox: Optional[Tuple[int, int, int, int]] = None  # bbox de la main (diagnostic)
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # bbox du visage ignoré (diagnostic)
    contour: object = None           # contour OpenCV de la main (pour affichage)
    hull: object = None              # enveloppe convexe (pour affichage)


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


# ======================================================================
# OpenCV Hand Detector — Détection robuste de la main via OpenCV
# ======================================================================

class OpenCVHandDetector:
    """Détecteur avancé de la main par OpenCV (anti-visage + longue distance).

    Résout deux limites majeures de la segmentation peau classique :
      1. Confusion visage / main : au lieu de prendre aveuglément le plus
         grand contour peau (souvent le visage), chaque contour est évalué par
         un score multi-critères combinant énergie de mouvement (absdiff EMA),
         solidité de l'enveloppe convexe (le visage est un ovale compact
         avec des trous sombres yeux/bouche), verrouillage temporel (Track
         Lock) et mémoire de zone d'exclusion visage.
      2. Perte à longue distance (3–5 m) : fonctionne en pleine résolution
         (downscale=1), utilise une morphologie douce (3x3, 1 itération) qui
         préserve les petites mains (~120 px²) et fusionne un seuillage peau
         élargi guidé par l'avant-plan MOG2 / mouvement.
    """

    def __init__(self, downscale=1, blur_k=3,
                 skin_ycrcb_min=(0, 133, 77), skin_ycrcb_max=(255, 173, 127),
                 skin_hsv_min=(0, 20, 35), skin_hsv_max=(25, 250, 255),
                 skin_hsv_min2=(165, 20, 35), skin_hsv_max2=(180, 250, 255),
                 open_kernel=3, close_kernel=5,
                 min_area=120, max_area_ratio=0.45,
                 pointer_point="fingertip",
                 use_bg_sub=True, bg_history=300, bg_threshold=25,
                 bg_detect_shadows=False, bg_learning_rate=0.002,
                 use_motion=True, face_reject=True,
                 **_ignored):
        if cv2 is None:
            raise ImportError("OpenCV est requis pour OpenCVHandDetector")

        self.downscale = max(1, int(downscale))
        self.blur_k = int(blur_k) | 1
        self.skin_ycrcb_min = np.array(skin_ycrcb_min, dtype=np.uint8)
        self.skin_ycrcb_max = np.array(skin_ycrcb_max, dtype=np.uint8)
        self.skin_hsv_min = np.array(skin_hsv_min, dtype=np.uint8)
        self.skin_hsv_max = np.array(skin_hsv_max, dtype=np.uint8)
        self.skin_hsv_min2 = np.array(skin_hsv_min2, dtype=np.uint8)
        self.skin_hsv_max2 = np.array(skin_hsv_max2, dtype=np.uint8)
        # Bornes élargies (utilisées UNIQUEMENT dans les zones en mouvement)
        self._relaxed_ycrcb_min = np.array([0, 128, 72], dtype=np.uint8)
        self._relaxed_ycrcb_max = np.array([255, 178, 133], dtype=np.uint8)
        self._relaxed_hsv_min = np.array([0, 15, 25], dtype=np.uint8)
        self._relaxed_hsv_max = np.array([28, 255, 255], dtype=np.uint8)

        self.open_kernel = int(open_kernel) | 1
        self.close_kernel = int(close_kernel) | 1
        self.min_area = int(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.pointer_point = pointer_point
        self.use_bg_sub = bool(use_bg_sub)
        self.bg_history = int(bg_history)
        self.bg_threshold = float(bg_threshold)
        self.bg_detect_shadows = bool(bg_detect_shadows)
        self.bg_learning_rate = float(bg_learning_rate)
        self.use_motion = bool(use_motion)
        self.face_reject = bool(face_reject)

        self._bg_sub = None
        if self.use_bg_sub:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.bg_history,
                varThreshold=self.bg_threshold,
                detectShadows=self.bg_detect_shadows,
            )

        self._prev_gray = None
        self._motion_ema = None
        self._last_pos = None          # (x, y) en résolution réduite (Track Lock)
        self._face_zone = None         # (x0, y0, x1, y1) en résolution réduite
        self._face_ttl = 0             # durée de vie de la zone d'exclusion visage
        self.last_mask = None
        self._last_contour = None
        self._last_hull = None
        self._last_face_bbox = None

    # ------------------------------------------------------------------
    def reset_background(self):
        """Réinitialise le modèle de fond MOG2 et la mémoire de mouvement."""
        if self._bg_sub is not None:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=self.bg_history,
                varThreshold=self.bg_threshold,
                detectShadows=self.bg_detect_shadows,
            )
        self._prev_gray = None
        self._motion_ema = None
        self._face_zone = None
        self._face_ttl = 0
        self._last_pos = None

    def set_pointer_mode(self, mode):
        """Bascule entre 'fingertip' et 'centroid'."""
        if mode in ("fingertip", "centroid"):
            self.pointer_point = mode

    def calibrate_skin_from_roi(self, frame_bgr, roi_box):
        """Auto-calibre les seuils peau YCrCb et HSV depuis un rectangle sur la main."""
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        h, w = frame_bgr.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return False
        patch = cv2.GaussianBlur(frame_bgr[y0:y1, x0:x1], (3, 3), 0)
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        cr_lo, cr_hi = np.percentile(ycrcb[:, 1], [5, 95])
        cb_lo, cb_hi = np.percentile(ycrcb[:, 2], [5, 95])
        self.skin_ycrcb_min = np.array([0, max(0, int(cr_lo) - 10), max(0, int(cb_lo) - 10)], dtype=np.uint8)
        self.skin_ycrcb_max = np.array([255, min(255, int(cr_hi) + 10), min(255, int(cb_hi) + 10)], dtype=np.uint8)

        s_lo = np.percentile(hsv[:, 1], 5)
        v_lo = np.percentile(hsv[:, 2], 5)
        self.skin_hsv_min[1] = max(10, int(s_lo) - 15)
        self.skin_hsv_min[2] = max(20, int(v_lo) - 20)
        self.skin_hsv_min2[1] = self.skin_hsv_min[1]
        self.skin_hsv_min2[2] = self.skin_hsv_min[2]
        return True

    # ------------------------------------------------------------------
    def _compute_skin_and_motion(self, frame_bgr):
        """Calcule les masques peau (strict + élargi) et l'énergie de mouvement."""
        f = self.downscale
        small = (cv2.resize(frame_bgr, None, fx=1.0 / f, fy=1.0 / f,
                            interpolation=cv2.INTER_AREA) if f > 1 else frame_bgr)

        blurred = cv2.GaussianBlur(small, (self.blur_k, self.blur_k), 0)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1) Masque peau strict (YCrCb ∩ HSV)
        m_ycrcb = cv2.inRange(ycrcb, self.skin_ycrcb_min, self.skin_ycrcb_max)
        m_hsv1 = cv2.inRange(hsv, self.skin_hsv_min, self.skin_hsv_max)
        m_hsv2 = cv2.inRange(hsv, self.skin_hsv_min2, self.skin_hsv_max2)
        strict_skin = cv2.bitwise_and(m_ycrcb, cv2.bitwise_or(m_hsv1, m_hsv2))

        # 2) Énergie de mouvement (absdiff EMA)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        if self.use_motion:
            if self._prev_gray is None or self._prev_gray.shape != gray.shape:
                self._prev_gray = gray
                self._motion_ema = np.zeros(gray.shape, dtype=np.float32)
            else:
                diff = cv2.absdiff(gray, self._prev_gray).astype(np.float32)
                self._prev_gray = gray
                self._motion_ema = 0.80 * self._motion_ema + 0.20 * diff
        else:
            self._motion_ema = np.zeros(gray.shape, dtype=np.float32)

        # 3) Avant-plan MOG2 + masque peau élargi dans les zones actives (longue distance)
        fg_active = None
        if self._bg_sub is not None:
            fg_raw = self._bg_sub.apply(small, learningRate=self.bg_learning_rate)
            fg_bin = (fg_raw > 128).astype(np.uint8) * 255
            fg_active = cv2.dilate(fg_bin, np.ones((5, 5), np.uint8), iterations=1)

        motion_active = (self._motion_ema > 6.0).astype(np.uint8) * 255
        if fg_active is not None:
            active_zone = cv2.bitwise_or(fg_active, motion_active)
            # Peau élargie uniquement dans l'avant-plan / mouvement
            r_ycrcb = cv2.inRange(ycrcb, self._relaxed_ycrcb_min, self._relaxed_ycrcb_max)
            r_hsv1 = cv2.inRange(hsv, self._relaxed_hsv_min, self._relaxed_hsv_max)
            r_hsv2 = cv2.inRange(hsv, np.array([162, 15, 25], dtype=np.uint8),
                                 np.array([180, 255, 255], dtype=np.uint8))
            relaxed_skin = cv2.bitwise_and(r_ycrcb, cv2.bitwise_or(r_hsv1, r_hsv2))
            skin_fg = cv2.bitwise_and(strict_skin, fg_active)
            relaxed_fg = cv2.bitwise_and(relaxed_skin, active_zone)
            skin = cv2.bitwise_or(skin_fg, relaxed_fg)
        else:
            # Sans MOG2 (ex: image unique) : peau stricte + peau élargie si mouvement
            if motion_active.any():
                r_ycrcb = cv2.inRange(ycrcb, self._relaxed_ycrcb_min, self._relaxed_ycrcb_max)
                r_hsv = cv2.inRange(hsv, self._relaxed_hsv_min, self._relaxed_hsv_max)
                relaxed_m = cv2.bitwise_and(cv2.bitwise_and(r_ycrcb, r_hsv), motion_active)
                skin = cv2.bitwise_or(strict_skin, relaxed_m)
            else:
                skin = strict_skin

        return skin, strict_skin, small

    # ------------------------------------------------------------------
    def _apply_morphology(self, mask):
        """Nettoyage morphologique doux (préserve les petites mains lointaines)."""
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
        return mask

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1, (bx1 - bx0) * (by1 - by0))
        return inter / float(area_a + area_b - inter)

    # ------------------------------------------------------------------
    def _select_hand_contour(self, clean_mask, raw_skin):
        """Évalue tous les contours peau et sépare la MAIN du VISAGE.

        Critères par candidat :
          - motion   : énergie moyenne de mouvement dans la bbox
          - solidity : area / hull_area (visage ≈ 0.88–0.98, main avec doigts < 0.88)
          - holes    : un visage a des trous sombres (yeux, sourcils, bouche) dans raw_skin
          - face_zone: mémoire spatiale du visage déjà détecté
          - track_lock: proximité avec la dernière position de la main
        """
        f = self.downscale
        frame_area = clean_mask.shape[0] * clean_mask.shape[1]
        min_a = self.min_area / (f * f)
        max_a = self.max_area_ratio * frame_area

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, None

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            bbox = (bx, by, bx + bw, by + bh)
            hull = cv2.convexHull(c)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = area / hull_area
            aspect = bw / float(max(1, bh))

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            motion_roi = self._motion_ema[by:by + bh, bx:bx + bw]
            motion_val = float(motion_roi.mean()) if motion_roi.size else 0.0

            # Trous sombres À L'INTÉRIEUR du contour (yeux, sourcils, bouche d'un visage)
            c_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(c_mask, [c - np.array([bx, by])], -1, 255, -1)
            inner = cv2.erode(c_mask, np.ones((3, 3), np.uint8)) > 0
            raw_roi = raw_skin[by:by + bh, bx:bx + bw]
            hole_ratio = float((raw_roi[inner] == 0).mean()) if inner.any() else 0.0

            candidates.append({
                "contour": c, "hull": hull, "area": area,
                "bbox": bbox, "cx": cx, "cy": cy,
                "solidity": solidity, "aspect": aspect,
                "motion": motion_val, "hole_ratio": hole_ratio,
            })

        if not candidates:
            return None, len(contours), None

        # Décroissance de la zone d'exclusion visage
        if self._face_ttl > 0:
            self._face_ttl -= 1
            if self._face_ttl == 0:
                self._face_zone = None

        max_motion = max(c["motion"] for c in candidates)
        detected_face_bbox = None

        if self.face_reject:
            for c in candidates:
                is_oval = (0.55 <= c["aspect"] <= 1.15) and (c["solidity"] >= 0.85)
                has_face_holes = c["hole_ratio"] > 0.03
                in_face_zone = (self._face_zone is not None and
                                self._bbox_iou(c["bbox"], self._face_zone) > 0.35)
                low_motion = (max_motion > 2.5 and c["motion"] < 0.45 * max_motion)

                c["face_like"] = False
                if len(candidates) >= 2:
                    if (is_oval and (low_motion or has_face_holes)) or (in_face_zone and low_motion):
                        c["face_like"] = True
                elif in_face_zone and c["motion"] < 2.0 and is_oval:
                    c["face_like"] = True

                if c["face_like"] and (detected_face_bbox is None or
                                        c["area"] > detected_face_bbox[4]):
                    detected_face_bbox = (*c["bbox"], c["area"])

        if detected_face_bbox is not None:
            fb = detected_face_bbox[:4]
            if self._face_zone is None:
                self._face_zone = fb
            else:
                self._face_zone = tuple(int(0.7 * o + 0.3 * n) for o, n in zip(self._face_zone, fb))
            self._face_ttl = 45

        # Calcul du score composite pour chaque candidat
        best = None
        best_score = -1.0
        for c in candidates:
            # Score de base = racine de l'aire (évite qu'un énorme visage écrase une petite main)
            score = float(np.sqrt(c["area"]))
            # Bonus fort pour le mouvement (la main qui pointe bouge)
            score *= (1.0 + 0.6 * min(c["motion"], 20.0))
            # Bonus pour une silhouette de main (doigts -> solidité < 0.88)
            if c["solidity"] < 0.88:
                score *= 1.35
            # Pénalité visage
            if c.get("face_like", False):
                score *= 0.05
            # Track Lock : continuité temporelle avec la position précédente
            if self._last_pos is not None:
                dist = float(np.hypot(c["cx"] - self._last_pos[0], c["cy"] - self._last_pos[1]))
                score *= (1.0 + 1.5 * float(np.exp(-(dist * dist) / (2.0 * 120.0 * 120.0))))

            if score > best_score:
                best_score = score
                best = c

        face_box_out = detected_face_bbox[:4] if detected_face_bbox is not None else self._face_zone
        return best, len(candidates), face_box_out

    # ------------------------------------------------------------------
    def _locate_pointer(self, cand):
        """Localise le point de pointage ('fingertip' ou 'centroid') sur le candidat."""
        contour = cand["contour"]
        hull = cand["hull"]
        cx, cy = cand["cx"], cand["cy"]
        self._last_hull = hull

        if self.pointer_point == "centroid":
            return cx, cy, hull, contour

        contour_pts = contour.reshape(-1, 2)
        if len(contour_pts) < 3:
            return cx, cy, hull, contour

        y_values = contour_pts[:, 1]
        y_min = y_values.min()
        y_range = y_values.max() - y_min
        if y_range < 1:
            return cx, cy, hull, contour

        y_thresh = y_min + y_range * 0.25
        top_points = contour_pts[y_values <= y_thresh]
        if len(top_points) == 0:
            return cx, cy, hull, contour

        return float(top_points[:, 0].mean()), float(top_points[:, 1].mean()), hull, contour

    # ------------------------------------------------------------------
    def detect(self, frame_bgr) -> Detection:
        """Détecte la main (sans confusion avec le visage, même à longue distance)."""
        f = self.downscale
        skin_mask, raw_skin, _ = self._compute_skin_and_motion(frame_bgr)
        clean_mask = self._apply_morphology(skin_mask)
        self.last_mask = clean_mask

        best, n_candidates, face_box = self._select_hand_contour(clean_mask, raw_skin)
        self._last_face_bbox = (
            (face_box[0] * f, face_box[1] * f, face_box[2] * f - 1, face_box[3] * f - 1)
            if face_box is not None else None
        )

        if best is None:
            self._last_contour = None
            self._last_hull = None
            return Detection(False, n_blobs=n_candidates,
                             reason="aucun contour de main valide", source="hand",
                             face_bbox=self._last_face_bbox)

        self._last_contour = best["contour"]
        px, py, hull, contour = self._locate_pointer(best)
        self._last_pos = (best["cx"], best["cy"])

        bx, by, bw, bh = cv2.boundingRect(contour)
        px_full = float(px * f + (f - 1) / 2.0)
        py_full = float(py * f + (f - 1) / 2.0)
        bbox_full = (bx * f, by * f, (bx + bw) * f - 1, (by + bh) * f - 1)
        area_full = int(best["area"] * f * f)

        return Detection(
            True, x=px_full, y=py_full, area=area_full,
            bbox=bbox_full, n_blobs=n_candidates,
            source="hand", hand_bbox=bbox_full,
            face_bbox=self._last_face_bbox,
            contour=contour, hull=hull,
        )
