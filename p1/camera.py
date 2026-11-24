"""
camera.py — Acquisition des images (OpenCV est utilisé ici, ce que le sujet autorise).

Points importants pour la cohérence avec P2 (calibration) :
  * la résolution utilisée pendant la calibration doit être EXACTEMENT celle
    utilisée pendant la détection (sinon l'homographie est fausse) ;
  * l'option `mirror` (config.json -> camera.mirror) retourne l'image
    horizontalement pour obtenir un effet miroir (main à droite -> pointeur à
    droite). Le retournement est fait ICI, une seule fois, donc détection,
    coordonnées et image de calibration sont tous dans le même repère.
    Si on change `mirror`, il faut REFAIRE la calibration de P2 ;
  * la caméra ne doit plus bouger ni zoomer après la calibration ;
  * on essaie de couper l'autofocus / l'exposition auto (pas toujours supporté).
"""

import platform
import time

import cv2

_BACKENDS = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,          # Windows (ouverture rapide)
    "msmf": cv2.CAP_MSMF,            # Windows (Media Foundation)
    "v4l2": cv2.CAP_V4L2,            # Linux
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
}


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=30, backend="auto",
                 autofocus=False, auto_exposure=True, loop_video=True,
                 mirror=False, **_ignored):
        # index peut être un entier (caméra) ou un chemin de vidéo (tests sans caméra)
        self.index = index
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.backend = backend
        self.autofocus = autofocus
        self.auto_exposure = auto_exposure
        self.loop_video = loop_video
        self.mirror = bool(mirror)       # True = effet miroir (flip horizontal)
        self.cap = None
        self.is_file = isinstance(index, str) and not index.isdigit()

    def open(self):
        src = self.index if self.is_file else int(self.index)
        be = self.backend
        if be == "auto" and not self.is_file and platform.system() == "Windows":
            be = "dshow"
        api = _BACKENDS.get(be)
        self.cap = cv2.VideoCapture(src) if (api is None or self.is_file) else cv2.VideoCapture(src, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra/vidéo : {self.index!r}. "
                               "Essayez un autre index (0, 1, 2) dans config.json.")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # moins de latence
            if not self.autofocus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if not self.auto_exposure:
                # valeur dépendante du pilote : 0.25 (V4L2) / 1 (DSHOW) = manuel
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # Lecture de contrôle + résolution RÉELLE (la caméra peut refuser la demande)
        ok, frame = False, None
        for _ in range(20):                                  # certaines webcams « chauffent »
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok:
            raise RuntimeError("La caméra est ouverte mais ne renvoie aucune image.")
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            print(f"[Camera] Attention : résolution demandée {self.width}x{self.height}, "
                  f"obtenue {w}x{h}. C'est {w}x{h} qui doit être utilisée par P2.")
        self.width, self.height = w, h
        return self

    def read(self):
        """Renvoie (ok, frame BGR uint8). Rejoue la vidéo en boucle si c'est un fichier.

        Si `mirror` est activé, l'image est retournée horizontalement (effet miroir).
        """
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and frame is not None and self.mirror:
            frame = cv2.flip(frame, 1)       # 1 = axe vertical -> gauche <-> droite
        return ok, frame

    @property
    def size(self):
        return self.width, self.height

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
