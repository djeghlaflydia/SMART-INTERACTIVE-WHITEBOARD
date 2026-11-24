"""
tune_hsv.py — Réglage des seuils HSV du marqueur, puis sauvegarde dans config.json.

Lancer :  python tools/tune_hsv.py        (depuis le dossier P1_detection)

Mode d'emploi :
  1. Tenez le marqueur devant le tableau, à l'endroit habituel.
  2. CLIQUEZ sur le marqueur dans la fenêtre « camera » : une plage HSV est
     calculée automatiquement autour de la couleur cliquée.
  3. Affinez avec les curseurs jusqu'à ce que le masque soit BLANC sur le
     marqueur et NOIR partout ailleurs (tableau, mains, vêtements, projection).
  4. Touche  s  = sauvegarder dans config.json     q / Échap = quitter
"""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from p1 import Camera, ColorMarkerDetector, load_config, save_config, sample_hsv_range  # noqa: E402

CONFIG = os.path.join(ROOT, "config.json")
NAMES = ["H min", "H max", "S min", "S max", "V min", "V max"]
MAXV = [179, 179, 255, 255, 255, 255]


def main():
    cfg = load_config(CONFIG)
    det = ColorMarkerDetector(**cfg["detection"])
    cam = Camera(**cfg["camera"]).open()

    cv2.namedWindow("camera")
    cv2.namedWindow("reglages", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("reglages", 400, 260)
    init = [det.hsv_lower[0], det.hsv_upper[0], det.hsv_lower[1],
            det.hsv_upper[1], det.hsv_lower[2], det.hsv_upper[2]]
    for name, mx, v in zip(NAMES, MAXV, init):
        cv2.createTrackbar(name, "reglages", int(v), mx, lambda _: None)

    last = {"frame": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and last["frame"] is not None:
            lo, up, med = sample_hsv_range(last["frame"], x, y)
            print(f"Clic ({x},{y}) : HSV médian = {med} -> plage {lo} .. {up}")
            for name, v in zip(NAMES, [lo[0], up[0], lo[1], up[1], lo[2], up[2]]):
                cv2.setTrackbarPos(name, "reglages", int(v))

    cv2.setMouseCallback("camera", on_mouse)
    print(__doc__)

    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        last["frame"] = frame
        v = [cv2.getTrackbarPos(n, "reglages") for n in NAMES]
        det.set_thresholds((v[0], v[2], v[4]), (v[1], v[3], v[5]))
        d = det.detect(frame)
        mask = det.last_mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        vis = frame.copy()
        if d.found:
            x0, y0, x1, y1 = d.bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.circle(vis, (int(d.x), int(d.y)), 5, (0, 0, 255), -1)
            txt = f"OK  ({d.x:.0f},{d.y:.0f})  aire={d.area}"
        else:
            txt = f"non detecte : {d.reason}"
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"lower={det.hsv_lower} upper={det.hsv_upper}  [s]=sauver [q]=quitter",
                    (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("camera", vis)
        cv2.imshow("masque", mask)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            cfg["detection"]["hsv_lower"] = list(det.hsv_lower)
            cfg["detection"]["hsv_upper"] = list(det.hsv_upper)
            save_config(cfg, CONFIG)
            print(f"Seuils sauvegardés dans {CONFIG} : {det.hsv_lower} .. {det.hsv_upper}")
        elif k in (ord("q"), 27):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
