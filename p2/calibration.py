"""
calibration.py
P2 - Selection et validation des quatre coins du tableau.
"""

import time

import cv2
import numpy as np

from .homography import CORNER_NAMES, order_corners, validate_points


class BoardCalibration:

    # Duree (secondes) d'affichage du message d'erreur avant de
    # recommencer automatiquement une selection invalide
    ERROR_DISPLAY_TIME = 2.0

    def __init__(self):
        self.points = []              # clics dans l'ordre ou ils sont faits
        self.window_name = "P2 - Calibration"
        self.error_message = ""
        self.error_time = None        # instant ou l'erreur est apparue
        self.image_size = None        # (largeur, hauteur) de l'image camera

    # ------------------------------------------------------------------
    # Souris
    # ------------------------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):
        """Enregistre les clics de souris (4 coins au maximum)."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"Coin {len(self.points)} selectionne : ({x}, {y})")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_selection(self):
        """
        Ordonne et valide les 4 points.
        Retourne les coins ordonnes (TL, TR, BR, BL) ou None si invalide
        (le message d'erreur est alors dans self.error_message).
        """

        try:
            ordered = order_corners(self.points)
            validate_points(ordered, self.image_size)
            self.error_message = ""
            return ordered

        except ValueError as error:
            self.error_message = str(error)
            return None

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_corners(self, tracker):
        """
        Selectionne les quatre coins du tableau.

        Les coins peuvent etre cliques dans n'importe quel ordre :
        ils sont ensuite ordonnes automatiquement (TL, TR, BR, BL).

        La selection est validee automatiquement des que les 4 coins
        forment un quadrilatere valide. Si elle est invalide, le message
        d'erreur s'affiche quelques secondes puis la selection est
        reinitialisee automatiquement.

        Touche Q : quitter.
        """

        self.points = []
        self.error_message = ""
        self.error_time = None

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== CALIBRATION P2 ===")
        print("Cliquez sur les 4 coins du tableau.")
        print("Touche Q : quitter\n")

        try:
            while True:

                frame = tracker.get_frame()

                if frame is None:
                    cv2.waitKey(10)
                    continue

                self.image_size = (frame.shape[1], frame.shape[0])

                display = frame.copy()
                ordered = None

                if len(self.points) == 4:
                    ordered = self._check_selection()

                    if ordered is None and self.error_time is None:
                        self.error_time = time.time()

                self._draw(display, ordered)

                cv2.imshow(self.window_name, display)

                # Selection valide : on la retourne
                if ordered is not None:
                    cv2.waitKey(300)   # laisse voir le resultat valide
                    print("Selection validee !")
                    return ordered

                # Selection invalide : reinitialisation automatique
                if (
                    self.error_time is not None
                    and time.time() - self.error_time > self.ERROR_DISPLAY_TIME
                ):
                    self.points = []
                    self.error_message = ""
                    self.error_time = None
                    print("Selection invalide : recommencez.")

                key = cv2.waitKey(20) & 0xFF

                if key == ord("q"):
                    self.points = []
                    print("Calibration annulee.")
                    return None

        finally:
            cv2.destroyWindow(self.window_name)

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def _draw(self, display, ordered):
        """Dessine les points, le quadrilatere et les messages."""

        if ordered is not None:
            # Selection valide : quadrilatere vert, coins nommes
            cv2.polylines(
                display,
                [ordered.astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )

            for name, (x, y) in zip(CORNER_NAMES, ordered):
                cv2.putText(
                    display,
                    name,
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                display,
                "Selection validee !",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif len(self.points) == 4:
            cv2.putText(
                display,
                "Selection invalide - recommencez",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Points cliques
        for i, (x, y) in enumerate(self.points):
            cv2.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)

            if ordered is None:
                cv2.putText(
                    display,
                    str(i + 1),
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Lignes entre les premiers points (avant le 4e clic)
        if 2 <= len(self.points) < 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 255), 2)

        # Message d'erreur
        if self.error_message:
            cv2.putText(
                display,
                self.error_message[:75],
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )