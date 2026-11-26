"""
test_p1.py — Tests automatiques de P1 (sans caméra).

Lancer :  python tests/test_p1.py     (depuis le dossier P1_detection)

1. Nos fonctions NumPy donnent le même résultat qu'OpenCV.
2. Le détecteur retrouve un marqueur synthétique à < 1 pixel près,
   même avec du bruit et une tache parasite.
3. Le détecteur ne « voit » rien quand le marqueur est absent.
4. Le stabilisateur réduit le tremblement et gère la disparition/réapparition.
5. Mesure du temps de traitement par image.
6. YCrCb NumPy vs OpenCV.
7. OpenCVHandDetector détecte une main synthétique et rejette les distracteurs.
"""

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from p1 import ColorMarkerDetector, Detection, PointerStabilizer  # noqa: E402
from p1 import image_ops as ops  # noqa: E402
from p1.detector import OpenCVHandDetector  # noqa: E402

rng = np.random.default_rng(0)
passed = failed = 0


def check(name, cond, info=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]    {name} {info}")
    else:
        failed += 1
        print(f"  [ECHEC] {name} {info}")


def scene(cx=None, cy=None, r=14, noise=6, distractor=False, w=640, h=480):
    """Image synthétique : tableau blanc légèrement dégradé + marqueur vert."""
    img = np.full((h, w, 3), 225, np.float32)
    img += np.linspace(-25, 15, w)[None, :, None]               # éclairage non uniforme
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 300), (200, 460), (150, 170, 210), -1)   # « main » couleur peau
    if cx is not None:
        cv2.circle(img, (int(round(cx)), int(round(cy))), r, (40, 190, 30), -1, cv2.LINE_AA)
    if distractor:
        cv2.circle(img, (580, 60), 2, (40, 190, 30), -1)          # petit point vert parasite
    img = img.astype(np.int16) + rng.normal(0, noise, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def hand_scene(hand_cx=320, hand_cy=200, hand_r=50,
               distractors=True, w=640, h=480):
    """Image synthétique avec une main (couleur peau) sur fond blanc/gris.

    La main est un large cercle couleur peau (BGR ≈ (120, 150, 200) → peau claire).
    Les distracteurs sont des rectangles de couleurs variées placés loin de la main.
    """
    # Fond : tableau blanc/gris légèrement dégradé
    img = np.full((h, w, 3), 220, np.float32)
    img += np.linspace(-15, 10, w)[None, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Main (peau) — couleur BGR typique de la peau humaine
    # BGR (120, 150, 200) → HSV ≈ H=15, S=100, V=200 → teinte peau
    # YCrCb ≈ Y=160, Cr=155, Cb=102 → dans la plage peau
    skin_bgr = (120, 150, 200)
    cv2.circle(img, (int(hand_cx), int(hand_cy)), hand_r, skin_bgr, -1, cv2.LINE_AA)

    # Bout du doigt (en haut de la main, pointant vers le haut)
    finger_tip_y = int(hand_cy - hand_r - 15)
    finger_tip_x = int(hand_cx)
    cv2.ellipse(img, (finger_tip_x, finger_tip_y + 10), (8, 20), 0, 0, 360, skin_bgr, -1)

    if distractors:
        # Distracteur 1 : rectangle noir (la télécommande du voisin !)
        cv2.rectangle(img, (550, 350), (600, 440), (30, 30, 30), -1)
        # Distracteur 2 : rectangle marron/bois (bureau)
        cv2.rectangle(img, (10, 400), (120, 470), (80, 120, 160), -1)
        # Distracteur 3 : petite tache couleur peau loin de la main
        cv2.circle(img, (580, 50), 8, skin_bgr, -1)

    # Bruit léger
    img = img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


print("\n1) Comparaison NumPy / OpenCV")
img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
hsv_np = ops.bgr_to_hsv(img).astype(int)
hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
dh = np.abs(hsv_np[..., 0] - hsv_cv[..., 0]); dh = np.minimum(dh, 180 - dh)
check("BGR->HSV (H)", dh.max() <= 1, f"écart max = {dh.max()}")
check("BGR->HSV (S,V)", np.abs(hsv_np[..., 1:] - hsv_cv[..., 1:]).max() <= 1,
      f"écart max = {np.abs(hsv_np[..., 1:] - hsv_cv[..., 1:]).max()}")

m = rng.random((100, 130)) > 0.6
k = np.ones((3, 3), np.uint8)
check("érosion", np.array_equal(ops.erode(m, 3), cv2.erode(m.astype(np.uint8), k) > 0))
check("dilatation", np.array_equal(ops.dilate(m, 3), cv2.dilate(m.astype(np.uint8), k) > 0))
k5 = np.ones((5, 5), np.uint8)
check("ouverture 5x5", np.array_equal(ops.opening(m, 5),
      cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k5) > 0))

bl_np = ops.box_blur(img, 3).astype(int)
bl_cv = cv2.blur(img, (3, 3), borderType=cv2.BORDER_REPLICATE).astype(int)
check("flou 3x3", np.abs(bl_np - bl_cv).max() <= 1, f"écart max = {np.abs(bl_np - bl_cv).max()}")

m2 = rng.random((90, 120)) > 0.7
lab, n = ops.connected_components(m2)
n_cv, lab_cv = cv2.connectedComponents(m2.astype(np.uint8), connectivity=8)
areas_np = sorted(np.bincount(lab.ravel())[1:].tolist())
areas_cv = sorted(np.bincount(lab_cv.ravel())[1:].tolist())
check("composantes connexes", n == n_cv - 1 and areas_np == areas_cv, f"{n} taches")

print("\n2) Détection d'un marqueur synthétique")
for use_cv in (False, True):
    det = ColorMarkerDetector(use_opencv=use_cv)
    errs = []
    for _ in range(20):
        cx, cy = rng.uniform(30, 610), rng.uniform(30, 290)
        d = det.detect(scene(cx, cy, distractor=True))
        errs.append(np.hypot(d.x - cx, d.y - cy) if d.found else 99)
    tag = "OpenCV" if use_cv else "NumPy "
    check(f"[{tag}] erreur de position < 1 px", max(errs) < 1.0,
          f"(moy {np.mean(errs):.2f}, max {max(errs):.2f} px)")

print("\n3) Absence de marqueur")
det = ColorMarkerDetector()
d = det.detect(scene(None, None, distractor=True))
check("aucune fausse détection", not d.found, f"raison : {d.reason}")

print("\n4) Stabilisateur")
stab = PointerStabilizer()
raw, smooth = [], []
for i in range(120):                                  # marqueur immobile + bruit
    x, y = 300 + rng.normal(0, 1.5), 200 + rng.normal(0, 1.5)
    s = stab.update(Detection(True, x=x, y=y, area=500), i)
    raw.append((x, y)); smooth.append((s.x, s.y))
raw, smooth = np.array(raw[20:]), np.array(smooth[20:])
sr = np.sqrt(raw.var(axis=0).sum()); ss = np.sqrt(smooth.var(axis=0).sum())
check("tremblement réduit", ss < sr / 2, f"(brut {sr:.2f} px -> lissé {ss:.2f} px)")

states = [stab.update(Detection(False), 200 + i).status for i in range(8)]
check("disparition : HOLD puis LOST", states[0] == "HOLD" and states[-1] == "LOST",
      str(states))
s = stab.update(Detection(True, x=500, y=100, area=500), 300)
check("réapparition immédiate à la nouvelle position",
      s.status == "TRACKING" and abs(s.x - 500) < 1e-6, f"({s.x:.1f}, {s.y:.1f})")

s1 = stab.update(Detection(True, x=50, y=400, area=500), 301)   # saut aberrant isolé
s2 = stab.update(Detection(True, x=501, y=101, area=500), 302)
check("saut aberrant isolé ignoré", abs(s1.x - 500) < 2 and abs(s2.x - 500) < 3,
      f"({s1.x:.1f} puis {s2.x:.1f})")

print("\n5) Temps de traitement (image 640x480)")
frame = scene(320, 200)
for use_cv in (False, True):
    det = ColorMarkerDetector(use_opencv=use_cv)
    det.detect(frame)
    t0 = time.perf_counter()
    for _ in range(30):
        det.detect(frame)
    ms = (time.perf_counter() - t0) / 30 * 1000
    tag = "OpenCV" if use_cv else "NumPy "
    check(f"[{tag}] temps par image", ms < 33, f"{ms:.1f} ms  (~{1000 / ms:.0f} FPS max)")

# ------------------------------------------------------------------
# Nouveaux tests : YCrCb et OpenCVHandDetector
# ------------------------------------------------------------------

print("\n6) YCrCb NumPy vs OpenCV")
img_ycr = rng.integers(0, 256, (100, 140, 3), dtype=np.uint8)
ycrcb_np = ops.bgr_to_ycrcb(img_ycr).astype(int)
ycrcb_cv = cv2.cvtColor(img_ycr, cv2.COLOR_BGR2YCrCb).astype(int)
diff_ycrcb = np.abs(ycrcb_np - ycrcb_cv)
check("BGR->YCrCb (Y,Cr,Cb)", diff_ycrcb.max() <= 2,
      f"écart max = {diff_ycrcb.max()}")

print("\n7) OpenCVHandDetector — détection de main synthétique")

# Test 7a : détection de base (main présente)
hand_det = OpenCVHandDetector(downscale=1, blur_k=3,
                               open_kernel=3, close_kernel=5,
                               min_area=200, max_area_ratio=0.5,
                               pointer_point="centroid",
                               use_bg_sub=False)
hand_img = hand_scene(hand_cx=320, hand_cy=200, hand_r=50, distractors=True)
d = hand_det.detect(hand_img)
if d.found:
    err = np.hypot(d.x - 320, d.y - 200)
    check("[Hand OpenCV] main détectée", True,
          f"pos=({d.x:.1f}, {d.y:.1f}), err={err:.1f} px, source={d.source}")
    # Le centroïde doit être raisonnablement proche du centre de la main
    check("[Hand OpenCV] erreur position < 30 px", err < 30,
          f"err = {err:.1f} px")
else:
    check("[Hand OpenCV] main détectée", False, f"raison : {d.reason}")
    check("[Hand OpenCV] erreur position < 30 px", False, "main non trouvée")

# Test 7b : mode fingertip (pointe le plus haut)
hand_det_tip = OpenCVHandDetector(downscale=1, blur_k=3,
                                   open_kernel=3, close_kernel=5,
                                   min_area=200, max_area_ratio=0.5,
                                   pointer_point="fingertip",
                                   use_bg_sub=False)
d_tip = hand_det_tip.detect(hand_img)
if d_tip.found:
    # Le bout du doigt doit être au-dessus du centroïde (y plus petit)
    check("[Hand OpenCV fingertip] doigt plus haut que centroïde",
          d_tip.y < d.y if d.found else True,
          f"tip_y={d_tip.y:.1f} < centroid_y={d.y:.1f}" if d.found else f"tip=({d_tip.x:.1f},{d_tip.y:.1f})")
else:
    check("[Hand OpenCV fingertip] doigt plus haut que centroïde", False,
          f"raison : {d_tip.reason}")

# Test 7c : pas de main (image vide)
empty = np.full((480, 640, 3), 220, np.uint8)
d_empty = hand_det.detect(empty)
check("[Hand OpenCV] aucune fausse détection sur fond vide", not d_empty.found,
      f"raison : {d_empty.reason}")

# Test 7d : champs source et hand_bbox
if d.found:
    check("[Hand OpenCV] source == 'hand'", d.source == "hand",
          f"source = {d.source}")
    check("[Hand OpenCV] hand_bbox renseigné", d.hand_bbox is not None,
          f"hand_bbox = {d.hand_bbox}")

# Test 7e : temps de traitement OpenCVHandDetector
print("\n8) Temps de traitement OpenCVHandDetector (640x480)")
hand_det_bench = OpenCVHandDetector(downscale=1, use_bg_sub=False)
hand_img_bench = hand_scene(320, 200, 50)
hand_det_bench.detect(hand_img_bench)  # warmup
t0 = time.perf_counter()
for _ in range(30):
    hand_det_bench.detect(hand_img_bench)
ms = (time.perf_counter() - t0) / 30 * 1000
check("[Hand OpenCV] temps par image", ms < 33,
      f"{ms:.1f} ms  (~{1000 / ms:.0f} FPS max)")

# ------------------------------------------------------------------
# Tests anti-visage, longue distance et filtre de Kalman
# ------------------------------------------------------------------
print("\n9) Anti-visage : main mobile vs grand visage ovale")
det_fv = OpenCVHandDetector(downscale=1, pointer_point="centroid", use_bg_sub=False)
skin_bgr = (120, 150, 200)
d_fv = None
for t in range(6):
    frame_fv = np.full((480, 640, 3), 220, np.uint8)
    # Grand visage ovale quasi-immobile avec yeux/bouche sombres
    cv2.ellipse(frame_fv, (160, 150), (55, 75), 0, 0, 360, skin_bgr, -1)
    cv2.circle(frame_fv, (140, 135), 9, (30, 30, 30), -1)
    cv2.circle(frame_fv, (180, 135), 9, (30, 30, 30), -1)
    cv2.ellipse(frame_fv, (160, 180), (20, 8), 0, 0, 360, (40, 40, 60), -1)
    # Petite main qui bouge à droite
    hx, hy = 400 + t * 18, 280 - t * 12
    cv2.circle(frame_fv, (hx, hy), 22, skin_bgr, -1)
    cv2.ellipse(frame_fv, (hx, hy - 26), (5, 12), 0, 0, 360, skin_bgr, -1)
    d_fv = det_fv.detect(frame_fv)

target_x, target_y = 400 + 5 * 18, 280 - 5 * 12
err_fv = np.hypot(d_fv.x - target_x, d_fv.y - target_y) if d_fv.found else 999
check("[Anti-Visage] verrouillé sur la main (et non le visage)",
      d_fv.found and err_fv < 30,
      f"pos=({d_fv.x:.1f}, {d_fv.y:.1f}) vs main=({target_x}, {target_y}), err={err_fv:.1f} px")

print("\n10) Longue distance : petite main (r=9, ~250 px²)")
det_ld = OpenCVHandDetector(downscale=1, pointer_point="centroid", use_bg_sub=False)
img_ld = hand_scene(hand_cx=480, hand_cy=160, hand_r=9, distractors=False)
d_ld = det_ld.detect(img_ld)
err_ld = np.hypot(d_ld.x - 480, d_ld.y - 160) if d_ld.found else 999
check("[Longue distance] petite main détectée (< 15 px)",
      d_ld.found and err_ld < 15,
      f"pos=({d_ld.x:.1f}, {d_ld.y:.1f}), err={err_ld:.1f} px, aire={d_ld.area}")

print("\n11) Stabilisateur Kalman 2D")
kf_stab = PointerStabilizer(filter_type="kalman")
raw_k, smooth_k = [], []
for i in range(120):
    x, y = 300 + rng.normal(0, 1.5), 200 + rng.normal(0, 1.5)
    s = kf_stab.update(Detection(True, x=x, y=y, area=300), i)
    raw_k.append((x, y)); smooth_k.append((s.x, s.y))
raw_k, smooth_k = np.array(raw_k[20:]), np.array(smooth_k[20:])
sr_k = np.sqrt(raw_k.var(axis=0).sum()); ss_k = np.sqrt(smooth_k.var(axis=0).sum())
check("[Kalman] tremblement réduit", ss_k < sr_k / 2,
      f"(brut {sr_k:.2f} px -> lissé {ss_k:.2f} px)")
k_states = [kf_stab.update(Detection(False), 200 + i).status for i in range(8)]
check("[Kalman] HOLD puis LOST", k_states[0] == "HOLD" and k_states[-1] == "LOST",
      str(k_states))

print(f"\nRésultat : {passed} réussis, {failed} échoués")
sys.exit(1 if failed else 0)
