"""
test_p1.py — Tests automatiques de P1 (sans caméra).

Lancer :  python tests/test_p1.py     (depuis le dossier P1_detection)

1. Nos fonctions NumPy donnent le même résultat qu'OpenCV.
2. Le détecteur retrouve un marqueur synthétique à < 1 pixel près,
   même avec du bruit et une tache parasite.
3. Le détecteur ne « voit » rien quand le marqueur est absent.
4. Le stabilisateur réduit le tremblement et gère la disparition/réapparition.
5. Mesure du temps de traitement par image.
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

print(f"\nRésultat : {passed} réussis, {failed} échoués")
sys.exit(1 if failed else 0)
