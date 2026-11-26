"""
demo_p1.py — Démonstration et TESTS de P1 avec la vraie caméra.

Lancer :  python tools/demo_p1.py   [--video chemin.mp4]   [--opencv]  [--method hand_opencv|color_marker]

Affiche le flux vidéo, le masque, la position détectée (brute + lissée),
la trajectoire, l'état (TRACKING / HOLD / LOST) et les FPS.

Touches :
  m  afficher / cacher le masque
  j  TEST DE STABILITÉ : gardez le marqueur immobile ~3 s ; affiche l'écart-type
     de la position brute et lissée (chiffres à mettre dans le rapport)
  l  journaliser dans logs/p1_log.csv (tests d'éclairage, disparition...)
  p  capture d'écran dans logs/ (pour le rapport)
  c  effacer la trajectoire
  b  réinitialiser le modèle de fond (soustraction de fond)
  a  auto-calibrer les seuils peau depuis le carré central
  f  basculer entre mode "fingertip" et "centroid" (hand_opencv uniquement)
  d  basculer entre hand_opencv et color_marker
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
from p1.detector import OpenCVHandDetector  # noqa: E402

COLORS = {"TRACKING": (0, 200, 0), "HOLD": (0, 165, 255), "LOST": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--video", help="vidéo à utiliser à la place de la caméra")
    ap.add_argument("--opencv", action="store_true", help="utiliser les fonctions OpenCV (comparaison)")
    ap.add_argument("--method", default=None, choices=["hand_opencv", "color_marker"],
                    help="méthode de détection")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["camera"]["index"] = args.video
    if args.opencv:
        cfg["detection"]["use_opencv"] = True
    if args.method:
        cfg["detection"]["method"] = args.method

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    tracker = PointerTracker(config=cfg).start()
    trail = deque(maxlen=60)
    show_mask = True
    jitter = None            # liste de positions pendant le test de stabilité
    log_file = log_writer = None
    last_idx = -1
    method_name = cfg["detection"].get("method", "color_marker")
    print(__doc__)
    print(f"Méthode active : {method_name}")

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

            # --- Dessin spécifique hand_opencv ---
            is_hand = isinstance(tracker.detector, OpenCVHandDetector)
            if is_hand:
                det = tracker.detector
                if det._last_face_bbox is not None:
                    fx0, fy0, fx1, fy1 = det._last_face_bbox
                    cv2.rectangle(vis, (fx0, fy0), (fx1, fy1), (0, 0, 220), 2)
                    cv2.putText(vis, "FACE IGNORED", (fx0, max(15, fy0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
                if st.detected:
                    f = det.downscale
                    if det._last_contour is not None:
                        contour_full = det._last_contour * f + (f - 1) // 2
                        cv2.drawContours(vis, [contour_full], -1, (0, 255, 0), 2)
                    if det._last_hull is not None:
                        hull_full = det._last_hull * f + (f - 1) // 2
                        cv2.drawContours(vis, [hull_full], -1, (255, 180, 0), 2)

            if st.detected and st.bbox:
                x0, y0, x1, y1 = st.bbox
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
                cv2.circle(vis, (int(st.raw_x), int(st.raw_y)), 3, (0, 0, 255), -1)   # brut
            if st.valid:
                cx, cy = int(st.x), int(st.y)                                         # lissé
                cv2.drawMarker(vis, (cx, cy), COLORS[st.status], cv2.MARKER_CROSS, 24, 2)

            # Infos texte
            source_tag = ""
            if is_hand:
                ptr_mode = tracker.detector.pointer_point
                source_tag = f"  [{st.source}|{ptr_mode}]"
            pos = f"({st.x:.1f}, {st.y:.1f})" if st.valid else "-"
            cv2.putText(vis, f"{st.status}  {pos}  aire={st.area}{source_tag}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLORS[st.status], 2)
            cv2.putText(vis, f"FPS {tracker.fps:.1f}  {st.frame_size[0]}x{st.frame_size[1]}"
                        f"  {method_name}",
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
                # Le masque hand_opencv est déjà uint8 (0 ou 255)
                mask_display = mask if mask.dtype == np.uint8 else mask.astype(np.uint8) * 255
                cv2.imshow("P1 - masque", mask_display)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("P1 - masque")
            elif k == ord("j"):
                jitter = []
                print("Test de stabilité : gardez le pointeur IMMOBILE...")
            elif k == ord("c"):
                trail.clear()
            elif k == ord("p"):
                name = os.path.join(ROOT, "logs", f"capture_{int(time.time())}.png")
                cv2.imwrite(name, vis)
                print("Capture :", name)
            elif k == ord("b"):
                tracker.reset_background()
                print("Modèle de fond réinitialisé.")
            elif k == ord("a"):
                if is_hand:
                    h, w = frame.shape[:2]
                    roi = (w // 2 - 40, h // 2 - 40, w // 2 + 40, h // 2 + 40)
                    if tracker.detector.calibrate_skin_from_roi(frame, roi):
                        print("Seuils peau auto-calibrés depuis le centre de l'image !")
            elif k == ord("f"):
                if is_hand:
                    cur = tracker.detector.pointer_point
                    new_mode = "centroid" if cur == "fingertip" else "fingertip"
                    tracker.detector.set_pointer_mode(new_mode)
                    print(f"Mode pointeur : {new_mode}")
                else:
                    print("Le mode fingertip/centroid n'est disponible qu'en hand_opencv.")
            elif k == ord("d"):
                # Hot-switch between methods (requires restart in practice)
                if method_name == "hand_opencv":
                    method_name = "color_marker"
                else:
                    method_name = "hand_opencv"
                print(f"Méthode basculée -> {method_name}. Redémarrage nécessaire.")
                print(f"  Modifier config.json -> detection.method = \"{method_name}\"")
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
