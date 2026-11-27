"""
test_p3_remote.py
Script de test matériel pour le module P3 (télécommande USB HID).

Permet de valider les 5 tests exigés par le cahier des charges P3 :
  1. Détectabilité de chaque bouton (Next et Previous).
  2. Distinction claire entre Next (LEFT_CLICK) et Previous (RIGHT_CLICK).
  3. Absence de déclenchement sans appui.
  4. Appuis rapides successifs.
  5. Fonctionnement après déconnexion et reconnexion du dongle USB.

Utilisation :
    python3 -m tools.test_p3_remote
"""

import cv2
import numpy as np

from p3.remote import LEFT_CLICK, RIGHT_CLICK, RemoteController


def main():
    remote = RemoteController.from_config("config.json")
    canvas = np.zeros((240, 620, 3), dtype=np.uint8)

    last_line = "En attente d'un appui (Next / Previous)..."
    color = (220, 220, 220)

    print("=== TEST MATÉRIEL P3 — TÉLÉCOMMANDE USB ===")
    print("1. Cliquez sur la fenêtre OpenCV qui vient de s'ouvrir.")
    print("2. Appuyez sur NEXT (Clic Gauche) ou PREVIOUS (Clic Droit).")
    print("3. Vous pouvez débrancher/rebrancher le dongle USB à tout moment.")
    print("4. Appuyez sur 'q' ou 'ESC' pour quitter.\n")

    cv2.namedWindow("P3 - Test Telecommande USB", cv2.WINDOW_AUTOSIZE)

    while True:
        frame = canvas.copy()
        stats = remote.get_stats()

        cv2.putText(
            frame,
            "P3 - Test Telecommande USB (Next / Previous)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (137, 180, 250),
            2,
        )
        cv2.putText(
            frame,
            last_line,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"NEXT (Gauche): {stats['total_next']}  |  "
            f"PREV (Droit): {stats['total_previous']}  |  "
            f"Debounce: {stats['ignored_debounce']}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            frame,
            "Touche 'q' ou ESC pour quitter",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (130, 130, 130),
            1,
        )

        cv2.imshow("P3 - Test Telecommande USB", frame)
        key = cv2.waitKeyEx(20)

        if key == -1:
            continue

        if key in (ord("q"), ord("Q"), 27):
            break

        event = remote.process_opencv_key(key)
        if event is not None:
            if event.action == LEFT_CLICK:
                last_line = f"[Code {event.raw_code}] NEXT -> CLIC GAUCHE"
                color = (80, 220, 120)
            elif event.action == RIGHT_CLICK:
                last_line = f"[Code {event.raw_code}] PREVIOUS -> CLIC DROIT"
                color = (255, 170, 60)
            print(last_line)
        else:
            print(f"[Code {key}] Touche ignoree ou filtree par anti-rebond.")

    cv2.destroyAllWindows()
    print("\nBilan P3 :", remote.get_stats())


if __name__ == "__main__":
    main()
