"""
check_camera.py — Étape 1 : trouver la caméra et vérifier l'image.

Lancer :  python tools/check_camera.py
  - teste les index 0 à 4 et affiche ceux qui fonctionnent (+ résolution) ;
  - ouvre ensuite le flux de l'index choisi dans config.json.
  q = quitter
"""

import os
import platform
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p1 import Camera, load_config  # noqa: E402

api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
print("Recherche des caméras...")
for i in range(5):
    cap = cv2.VideoCapture(i, api)
    if cap.isOpened():
        ok, f = cap.read()
        if ok:
            print(f"  index {i} : OK, {f.shape[1]}x{f.shape[0]}")
    cap.release()

cfg = load_config(os.path.join(ROOT, "config.json"))
cam = Camera(**cfg["camera"]).open()
print(f"Flux de l'index {cfg['camera']['index']} en {cam.width}x{cam.height}  (q pour quitter)")
t0, n = time.time(), 0
while True:
    ok, frame = cam.read()
    if not ok:
        continue
    n += 1
    fps = n / (time.time() - t0)
    cv2.putText(frame, f"{cam.width}x{cam.height}  {fps:.1f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("check camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
