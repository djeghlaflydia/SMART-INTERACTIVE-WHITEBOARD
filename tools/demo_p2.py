"""
demo_p2.py
Demonstration de P2 : calibration, homographie et transformation
de la position du pointeur detecte par P1.

Utilisation :
    python demo_p2.py

Etapes :
    1. Cliquez sur les 4 coins du tableau (la selection est validee
       automatiquement).
    2. Le pointeur detecte par P1 est transforme en coordonnees tableau.

Touche Q : quitter.
"""

import cv2
import numpy as np

from p1.tracker import PointerTracker
from p2.calibration import BoardCalibration
from p2.transformer import BoardTransformer

BOARD_WIDTH = 1280     # doit correspondre a la fenetre projetee de P5
BOARD_HEIGHT = 720

# Fenetre "vue redressee" : utile pour verifier la perspective, mais
# couteuse en calcul (mettre True seulement pour la verification).
SHOW_RECTIFIED_VIEW = False


def draw_overlay(frame, transformer):
    """Dessine le contour du tableau calibre sur l'image camera."""

    cv2.polylines(
        frame,
        [transformer.camera_points.astype(np.int32)],
        True,
        (255, 200, 0),
        2
    )


def show_rectified_view(frame, transformer):
    """
    Affiche le tableau 'redresse' pour verifier visuellement que la
    perspective est corrigee. (cv2.warpPerspective : affichage seulement,
    le calcul de la matrice est fait par notre code NumPy.)
    """

    width, height = transformer.board_size

    warped = cv2.warpPerspective(
        frame,
        transformer.get_matrix(),
        (width, height)
    )

    cv2.imshow(
        "P2 - Vue redressee (verification)",
        cv2.resize(warped, (width // 2, height // 2))
    )


def main():
    tracker = PointerTracker("config.json")

    try:
        # 1. Demarrer la camera
        tracker.start()
        print("Camera demarree.")

        # 2. Selectionner les quatre coins du tableau
        calibration = BoardCalibration()
        camera_points = calibration.select_corners(tracker)

        if camera_points is None:
            print("Calibration annulee.")
            return

        print("\nCoins selectionnes (TL, TR, BR, BL) :")
        for i, point in enumerate(camera_points):
            print(f"Coin {i + 1} : {point}")

        # 3. Calibrer le tableau
        transformer = BoardTransformer(
            board_width=BOARD_WIDTH,
            board_height=BOARD_HEIGHT
        )

        transformer.calibrate(camera_points, calibration.image_size)

        print("\nMatrice d'homographie :")
        print(transformer.get_matrix())
        print(
            f"\nErreur de reprojection des coins : "
            f"{transformer.verify():.2e} px"
        )
        print("\nCalibration terminee !")

        # 4. Tester la transformation en temps reel
        print("\nAppuyez sur Q pour quitter.")

        while True:
            frame = tracker.get_frame()

            if frame is None:
                cv2.waitKey(1)
                continue

            draw_overlay(frame, transformer)

            if SHOW_RECTIFIED_VIEW:
                show_rectified_view(frame, transformer)

            state = tracker.get_pointer()
            position = state.position()

            if position is not None:
                x, y = position

                try:
                    board_x, board_y = transformer.transform(x, y)
                    inside = transformer.is_inside(board_x, board_y)

                    # Point detecte par P1
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)

                    # Coordonnees transformees
                    text = f"Board: ({board_x:.0f}, {board_y:.0f})"
                    if not inside:
                        text += " - hors tableau"

                    cv2.putText(
                        frame,
                        text,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if inside else (0, 0, 255),
                        2
                    )

                except ValueError:
                    # Point invalide (ex. a l'infini) : on l'ignore
                    pass

            cv2.imshow("P2 - Calibration et Homographie", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Erreur : {e}")

    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()