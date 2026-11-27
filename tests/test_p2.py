"""
test_p2.py
Tests de P2 (sans camera). Lancer avec :  python test_p2.py

Principe : on simule une camera qui voit le tableau en perspective
avec une homographie connue, puis on verifie que notre calibration
retrouve bien les coordonnees du tableau.
"""

import numpy as np

from p2.homography import (
    HomographyTransformer,
    apply_homography,
    compute_homography,
    order_corners,
    validate_points,
)
from p2.transformer import BoardTransformer

W, H = 1280, 720

# Coins du tableau vus par la camera (perspective : haut plus etroit)
CAMERA_CORNERS = np.array([
    [210, 120],    # TL
    [520, 100],    # TR
    [560, 330],    # BR
    [180, 340],    # BL
], dtype=np.float32)


def intersection_diagonales(p):
    """Point d'intersection des diagonales TL-BR et TR-BL."""
    p = np.asarray(p, dtype=np.float64)   # float64 : precision du test
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p[0], p[2], p[1], p[3]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def test_coins():
    """Les 4 coins camera doivent tomber sur les 4 coins du tableau."""
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)

    attendu = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
    for (x, y), (u, v) in zip(CAMERA_CORNERS, attendu):
        bx, by = t.transform(x, y)
        assert abs(bx - u) < 1e-6 and abs(by - v) < 1e-6, (bx, by, u, v)

    assert t.verify() < 1e-6
    print("OK  correspondance des 4 coins")


def test_centre():
    """
    En perspective, le centre du tableau est l'intersection des
    diagonales du quadrilatere (pas la moyenne des 4 coins).
    """
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)

    cx, cy = intersection_diagonales(CAMERA_CORNERS)
    bx, by = t.transform(cx, cy)

    assert abs(bx - (W - 1) / 2) < 1e-6
    assert abs(by - (H - 1) / 2) < 1e-6
    print("OK  centre du tableau")


def test_zones():
    """Plusieurs zones : on simule la camera avec une H connue."""
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)

    # Matrice inverse : tableau -> camera, pour generer des points test
    inverse = np.linalg.inv(t.get_matrix())

    zones = [(100, 100), (1100, 100), (640, 360),
             (100, 600), (1100, 600), (320, 180)]
    for u, v in zones:
        x, y = apply_homography(inverse, u, v)
        bx, by = t.transform(x, y)
        assert abs(bx - u) < 1e-6 and abs(by - v) < 1e-6
    print("OK  differentes zones du tableau")


def test_perspective_corrigee():
    """Des points alignes sur le tableau restent alignes apres H."""
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)
    inverse = np.linalg.inv(t.get_matrix())

    # Ligne horizontale y = 360 sur le tableau, vue en perspective
    ys = []
    for u in np.linspace(0, W - 1, 9):
        x, y = apply_homography(inverse, u, 360)
        ys.append(t.transform(x, y)[1])

    assert np.allclose(ys, 360, atol=1e-6)
    print("OK  la perspective est corrigee")


def test_limites():
    """Dedans / dehors / clamp."""
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)

    assert t.is_inside(0, 0) and t.is_inside(W - 1, H - 1)
    assert not t.is_inside(-1, 10) and not t.is_inside(W, 10)

    # Point camera loin a gauche du tableau
    bx, by, inside = t.transform_clamped(5, 200)
    assert not inside and 0 <= bx <= W - 1 and 0 <= by <= H - 1
    print("OK  limites du tableau et clamp")


def test_comparaison_opencv():
    """Notre matrice doit etre identique a celle d'OpenCV."""
    try:
        import cv2   # noqa: F401
    except ImportError:
        print("--  OpenCV absent : comparaison ignoree")
        return

    h = HomographyTransformer(W, H)
    h.calculate(CAMERA_CORNERS)
    ecart = h.compare_with_opencv(CAMERA_CORNERS)

    assert ecart < 1e-4, ecart
    print(f"OK  identique a OpenCV (ecart max = {ecart:.2e})")


def test_ordre_automatique():
    """Les coins melanges sont remis dans l'ordre TL, TR, BR, BL."""
    melange = CAMERA_CORNERS[[2, 0, 3, 1]]
    ordonne = order_corners(melange)

    assert np.allclose(ordonne, CAMERA_CORNERS)
    print("OK  ordre automatique des coins")


def test_selections_invalides():
    """Les mauvaises selections doivent etre refusees."""
    invalides = {
        "points identiques": [[10, 10], [10, 10], [200, 200], [10, 200]],
        "sens inverse (miroir)": CAMERA_CORNERS[[0, 3, 2, 1]],
        "quadrilatere croise": CAMERA_CORNERS[[0, 2, 1, 3]],
        "points alignes": [[0, 0], [100, 0], [200, 0], [300, 0]],
        "trop petit": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "NaN": [[0, 0], [np.nan, 0], [8, 8], [0, 8]],
    }

    for nom, points in invalides.items():
        try:
            validate_points(points)
        except ValueError:
            continue
        raise AssertionError(f"Selection invalide acceptee : {nom}")

    # Points alignes : compute_homography doit aussi refuser
    try:
        compute_homography(
            [[0, 0], [100, 0], [200, 0], [300, 0]],
            [[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        raise AssertionError("Homographie degeneree acceptee")
    except ValueError:
        pass

    print("OK  selections invalides refusees")


def test_etat_apres_echec():
    """Une calibration qui echoue ne doit pas laisser l'ancienne active."""
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)

    try:
        t.calibrate([[0, 0], [1, 1], [2, 2], [3, 3]])
    except ValueError:
        pass

    assert not t.is_calibrated
    try:
        t.transform(100, 100)
        raise AssertionError("transform() accepte sans calibration")
    except RuntimeError:
        pass
    print("OK  etat coherent apres une calibration echouee")


if __name__ == "__main__":
    test_coins()
    test_centre()
    test_zones()
    test_perspective_corrigee()
    test_limites()
    test_comparaison_opencv()
    test_ordre_automatique()
    test_selections_invalides()
    test_etat_apres_echec()
    print("\nTous les tests P2 sont passes.")