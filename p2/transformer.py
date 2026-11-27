"""
transformer.py
P2 - Interface utilisee par les autres modules (P4, P5) pour convertir
les coordonnees camera en coordonnees tableau.
"""

import numpy as np

from .homography import HomographyTransformer


class BoardTransformer:
    """
    Transforme des coordonnees camera en coordonnees logiques du tableau.

    Le repere tableau a son origine en haut a gauche, x vers la droite,
    y vers le bas. Ses dimensions (board_width x board_height) doivent
    correspondre a la fenetre projetee par P5.
    """

    def __init__(self, board_width=1280, board_height=720):
        self.homography = HomographyTransformer(
            board_width=board_width,
            board_height=board_height
        )

        self.camera_points = None
        self.is_calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, camera_points, image_size=None):
        """
        Calcule l'homographie a partir des quatre coins camera.
        Ordre : TL, TR, BR, BL.
        """

        # Si le calcul echoue, on ne garde pas l'ancienne calibration
        self.is_calibrated = False
        self.camera_points = None

        self.homography.calculate(camera_points, image_size)

        self.camera_points = np.asarray(camera_points, dtype=np.float32)
        self.is_calibrated = True

    def verify(self):
        """
        Verifie la precision de la calibration.
        Retourne l'erreur maximale de reprojection des 4 coins (pixels).
        """
        self._check_calibrated()

        return self.homography.reprojection_error(self.camera_points)

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, x, y):
        """Transforme un point camera en coordonnees tableau."""
        self._check_calibrated()

        return self.homography.transform_point(x, y)

    def transform_clamped(self, x, y):
        """
        Transforme un point camera et ramene le resultat dans les
        limites du tableau. Retourne (board_x, board_y, etait_dedans).
        """
        board_x, board_y = self.transform(x, y)
        inside = self.homography.is_inside_board(board_x, board_y)
        board_x, board_y = self.homography.clamp_to_board(board_x, board_y)

        return board_x, board_y, inside

    def is_inside(self, x, y):
        """Verifie si des coordonnees tableau sont dans le tableau."""
        self._check_calibrated()

        return self.homography.is_inside_board(x, y)

    def get_matrix(self):
        """Retourne la matrice d'homographie."""
        self._check_calibrated()

        return self.homography.get_matrix()

    @property
    def board_size(self):
        return self.homography.board_width, self.homography.board_height

    # ------------------------------------------------------------------

    def _check_calibrated(self):
        if not self.is_calibrated:
            raise RuntimeError(
                "Le tableau doit etre calibre avant utilisation."
            )