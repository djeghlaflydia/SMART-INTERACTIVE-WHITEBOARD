"""
demo_p1.py — Démonstration et TESTS de P1 avec la vraie caméra.

Lancer :  python tools/demo_p1.py   [--video chemin.mp4]   [--opencv]

Affiche le flux vidéo, le masque, la position détectée (brute + lissée),
la trajectoire, l'état (TRACKING / HOLD / LOST) et les FPS.

Touches :
  m  afficher / cacher le masque
  j  TEST DE STABILITÉ : gardez le marqueur immobile ~3 s ; affiche l'écart-type
     de la position brute et lissée (chiffres à mettre dans le rapport)
  l  journaliser dans logs/p1_log.csv (tests d'éclairage, disparition...)
  p  capture d'écran dans logs/ (pour le rapport)
  c  effacer la trajectoire
  q  quitter
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from p1 import PointerTracker, load_config  # noqa: E402

COLORS = {"TRACKING": (0, 200, 0), "HOLD": (0, 165, 255), "LOST": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--video", help="vidéo à utiliser à la place de la caméra")
    ap.add_argument("--opencv", action="store_true", help="utiliser les fonctions OpenCV (comparaison)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["camera"]["index"] = args.video
    if args.opencv:
        cfg["detection"]["use_opencv"] = True

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    tracker = PointerTracker(config=cfg).start()
    trail = deque(maxlen=60)
    show_mask = True
    jitter = None            # liste de positions pendant le test de stabilité
    log_file = log_writer = None
    last_idx = -1
    print(__doc__)

    try:
        while True:
            frame, mask, st = tracker.get_debug()
            if frame is None or st.frame_index == last_idx:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_idx = st.frame_index
            vis = frame.copy()

            if st.valid:
                trail.append((int(st.x), int(st.y)))
            for i in range(1, len(trail)):
                cv2.line(vis, trail[i - 1], trail[i], (255, 0, 255), 2)
            if st.detected and st.bbox:
                x0, y0, x1, y1 = st.bbox
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
                cv2.circle(vis, (int(st.raw_x), int(st.raw_y)), 3, (0, 0, 255), -1)   # brut
            if st.valid:
                cx, cy = int(st.x), int(st.y)                                         # lissé
                cv2.drawMarker(vis, (cx, cy), COLORS[st.status], cv2.MARKER_CROSS, 24, 2)

            pos = f"({st.x:.1f}, {st.y:.1f})" if st.valid else "-"
            cv2.putText(vis, f"{st.status}  {pos}  aire={st.area}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[st.status], 2)
            cv2.putText(vis, f"FPS {tracker.fps:.1f}  {st.frame_size[0]}x{st.frame_size[1]}"
                        f"  {'OpenCV' if tracker.detector.use_opencv else 'NumPy'}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # --- test de stabilité ---
            if jitter is not None:
                if st.detected:
                    jitter.append((st.raw_x, st.raw_y, st.x, st.y))
                cv2.putText(vis, f"TEST STABILITE... {len(jitter)}/90", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if len(jitter) >= 90:
                    a = np.array(jitter)
                    sr = np.sqrt(a[:, 0].var() + a[:, 1].var())
                    ss = np.sqrt(a[:, 2].var() + a[:, 3].var())
                    print(f"[Stabilité] écart-type brut = {sr:.2f} px | lissé = {ss:.2f} px | "
                          f"amplitude lissée = {np.ptp(a[:, 2]):.1f} x {np.ptp(a[:, 3]):.1f} px")
                    jitter = None

            # --- journal CSV ---
            if log_writer:
                log_writer.writerow([f"{st.timestamp:.3f}", st.frame_index, st.status,
                                     f"{st.raw_x:.2f}", f"{st.raw_y:.2f}",
                                     f"{st.x:.2f}", f"{st.y:.2f}", st.area, f"{tracker.fps:.1f}"])
                cv2.circle(vis, (vis.shape[1] - 20, 20), 8, (0, 0, 255), -1)

            cv2.imshow("P1 - camera", vis)
            if show_mask and mask is not None:
                cv2.imshow("P1 - masque", mask.astype(np.uint8) * 255)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("P1 - masque")
            elif k == ord("j"):
                jitter = []
                print("Test de stabilité : gardez le marqueur IMMOBILE...")
            elif k == ord("c"):
                trail.clear()
            elif k == ord("p"):
                name = os.path.join(ROOT, "logs", f"capture_{int(time.time())}.png")
                cv2.imwrite(name, vis)
                print("Capture :", name)
            elif k == ord("l"):
                if log_writer:
                    log_file.close()
                    log_file = log_writer = None
                    print("Journal arrêté.")
                else:
                    path = os.path.join(ROOT, "logs", "p1_log.csv")
                    log_file = open(path, "w", newline="", encoding="utf-8")
                    log_writer = csv.writer(log_file)
                    log_writer.writerow(["t", "frame", "status", "raw_x", "raw_y",
                                         "x", "y", "area", "fps"])
                    print("Journal démarré :", path)
    finally:
        if log_file:
            log_file.close()
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
