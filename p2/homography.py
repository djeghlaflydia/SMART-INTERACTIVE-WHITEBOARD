"""
homography.py
P2 - Calcul et application de l'homographie (implementation NumPy).

Repere camera  : origine en haut a gauche de l'image, x vers la droite,
                 y vers le bas (en pixels).
Repere tableau : origine au coin haut gauche du tableau, x vers la droite,
                 y vers le bas, dimensions board_width x board_height.

Les etapes essentielles (ordre des coins, validation du quadrilatere,
calcul de la matrice H, application de H) 
"""

import numpy as np

# Noms des coins, dans l'ordre attendu (sens horaire a l'ecran)
CORNER_NAMES = ("TL", "TR", "BR", "BL")


# ----------------------------------------------------------------------
# Fonctions geometriques de base
# ----------------------------------------------------------------------

def order_corners(points):
    """
    Ordonne 4 points dans l'ordre TL, TR, BR, BL.

    Principe (axes image : x vers la droite, y vers le bas) :
      - TL a la plus petite somme x + y
      - BR a la plus grande somme x + y
      - TR a la plus petite difference y - x
      - BL a la plus grande difference y - x
    """
    pts = np.asarray(points, dtype=np.float64)

    if pts.shape != (4, 2):
        raise ValueError("Il faut exactement 4 points (x, y).")

    somme = pts[:, 0] + pts[:, 1]
    diff = pts[:, 1] - pts[:, 0]

    indices = [
        int(np.argmin(somme)),   # TL
        int(np.argmin(diff)),    # TR
        int(np.argmax(somme)),   # BR
        int(np.argmax(diff)),    # BL
    ]

    if len(set(indices)) != 4:
        raise ValueError(
            "Impossible d'ordonner les coins : "
            "les points ne forment pas un quadrilatere exploitable."
        )

    return pts[indices].astype(np.float32)


def signed_area(points):
    """
    Aire orientee du polygone (formule du lacet / shoelace).

    Avec l'axe y vers le bas, l'ordre TL -> TR -> BR -> BL
    (sens horaire a l'ecran) donne une aire POSITIVE.
    """
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]

    return 0.5 * float(
        np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )


def is_convex_clockwise(points):
    """
    Verifie que le quadrilatere est convexe ET parcouru dans le bon sens.

    Pour chaque sommet, on calcule le produit vectoriel de l'arete
    qui arrive et de l'arete qui part. Si les 4 produits sont tous
    strictement positifs : convexe, sans croisement, sens TL,TR,BR,BL.
    """
    pts = np.asarray(points, dtype=np.float64)

    for i in range(4):
        a = pts[(i + 1) % 4] - pts[i]
        b = pts[(i + 2) % 4] - pts[(i + 1) % 4]

        cross = a[0] * b[1] - a[1] * b[0]

        if cross <= 1e-9:
            return False

    return True


def validate_points(camera_points, image_size=None):
    """
    Verifie que 4 points forment un quadrilatere valide.
    Ordre attendu : TL, TR, BR, BL.

    image_size = (largeur, hauteur) de l'image camera (optionnel).
    S'il est donne, les seuils sont relatifs a la taille de l'image.
    """
    points = np.asarray(camera_points, dtype=np.float32)

    if points.shape != (4, 2):
        raise ValueError("Il faut exactement 4 points (x, y).")

    if not np.all(np.isfinite(points)):
        raise ValueError("Les coordonnees doivent etre finies.")

    # Seuils : relatifs a l'image si on la connait
    if image_size is not None:
        largeur, hauteur = image_size
        distance_min = max(5.0, 0.02 * min(largeur, hauteur))
        aire_min = 0.01 * largeur * hauteur
    else:
        distance_min = 5.0
        aire_min = 100.0

    # 1. Points distincts
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :],
        axis=2
    )
    distances += np.eye(4) * 1e9

    if np.min(distances) < distance_min:
        raise ValueError("Certains coins sont trop proches ou identiques.")

    # 2. Quadrilatere convexe + bon sens de parcours
    if not is_convex_clockwise(points):
        raise ValueError(
            "Les coins ne forment pas un quadrilatere convexe "
            "dans l'ordre TL, TR, BR, BL."
        )

    # 3. Surface suffisante
    if signed_area(points) < aire_min:
        raise ValueError("Le tableau selectionne est trop petit.")

    return points


# ----------------------------------------------------------------------
# Calcul et application de l'homographie
# ----------------------------------------------------------------------

def compute_homography(src, dst):
    """
    Calcule la matrice H (3x3) telle que  dst ~ H @ src,
    a partir de 4 correspondances de points.

    Pour un point (x, y) -> (u, v), avec h33 = 1 :
        u = (h1 x + h2 y + h3) / (h7 x + h8 y + 1)
        v = (h4 x + h5 y + h6) / (h7 x + h8 y + 1)
    En multipliant par le denominateur, chaque point donne 2 equations
    LINEAIRES en h1..h8 :
        h1 x + h2 y + h3 - h7 u x - h8 u y = u
        h4 x + h5 y + h6 - h7 v x - h8 v y = v
    4 points -> 8 equations, 8 inconnues : systeme 8x8 resolu par NumPy.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Il faut 4 points source et 4 points destination.")

    A = []
    b = []

    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)

    try:
        h = np.linalg.solve(np.array(A), np.array(b))
    except np.linalg.LinAlgError:
        raise ValueError(
            "Homographie impossible : points alignes ou degeneres."
        )

    matrix = np.append(h, 1.0).reshape(3, 3)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homographie invalide.")

    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Matrice d'homographie singuliere.")

    return matrix


def apply_homography(matrix, x, y):
    """
    Applique H a un point (x, y) :
      1. coordonnees homogenes  p = [x, y, 1]
      2. produit matriciel      q = H @ p
      3. division par w = q[2]  -> (q[0] / w, q[1] / w)
    """
    q = matrix @ np.array([x, y, 1.0])

    if abs(q[2]) < 1e-12:
        raise ValueError("Le point est a l'infini apres transformation.")

    return float(q[0] / q[2]), float(q[1] / q[2])


# ----------------------------------------------------------------------
# Classe principale
# ----------------------------------------------------------------------

class HomographyTransformer:

    def __init__(self, board_width=1280, board_height=720):

        if board_width <= 1 or board_height <= 1:
            raise ValueError(
                "Les dimensions du tableau doivent etre superieures a 1."
            )

        self.board_width = board_width
        self.board_height = board_height
        self.matrix = None

        # Coins du tableau dans le repere de destination
        self.destination_points = np.array([
            [0, 0],
            [board_width - 1, 0],
            [board_width - 1, board_height - 1],
            [0, board_height - 1]
        ], dtype=np.float32)

    def validate_points(self, camera_points, image_size=None):
        """Voir validate_points() du module."""
        return validate_points(camera_points, image_size)

    def calculate(self, camera_points, image_size=None):
        """
        Calcule la matrice d'homographie camera -> tableau
        a partir des quatre coins (TL, TR, BR, BL) de la camera.
        """

        # Une nouvelle calibration invalide l'ancienne matrice
        self.matrix = None

        camera_points = validate_points(camera_points, image_size)

        self.matrix = compute_homography(
            camera_points,
            self.destination_points
        )

        return self.matrix

    def transform_point(self, x, y):
        """Transforme un point de l'image camera vers le repere tableau."""

        if self.matrix is None:
            raise RuntimeError("Calculez d'abord l'homographie.")

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Les coordonnees du point doivent etre finies.")

        return apply_homography(self.matrix, x, y)

    def is_inside_board(self, x, y):
        """Verifie si un point transforme appartient au tableau."""

        return bool(
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x <= self.board_width - 1
            and 0 <= y <= self.board_height - 1
        )

    def clamp_to_board(self, x, y):
        """Ramene un point dans les limites du tableau."""

        x = min(max(x, 0.0), self.board_width - 1.0)
        y = min(max(y, 0.0), self.board_height - 1.0)

        return float(x), float(y)

    def reprojection_error(self, camera_points):
        """
        Verification de la precision : transforme les 4 coins camera
        et mesure l'ecart (en pixels tableau) avec les coins attendus.
        Retourne l'erreur maximale. Doit etre ~0 (< 1e-6).
        """

        if self.matrix is None:
            raise RuntimeError("Calculez d'abord l'homographie.")

        erreurs = []

        for (x, y), (u, v) in zip(camera_points, self.destination_points):
            tx, ty = apply_homography(self.matrix, x, y)
            erreurs.append(np.hypot(tx - u, ty - v))

        return float(max(erreurs))

    def compare_with_opencv(self, camera_points):
        """
        Compare notre matrice avec cv2.getPerspectiveTransform.
        Retourne l'ecart maximal entre les deux matrices
        (utile pour les tests et le rapport).
        """
        import cv2   # utilise uniquement pour verification

        if self.matrix is None:
            raise RuntimeError("Calculez d'abord l'homographie.")

        reference = cv2.getPerspectiveTransform(
            np.asarray(camera_points, dtype=np.float32),
            self.destination_points
        )

        reference = reference / reference[2, 2]

        return float(np.max(np.abs(reference - self.matrix)))

    def get_matrix(self):
        """Retourne une copie de la matrice d'homographie."""

        if self.matrix is None:
            return None

        return self.matrix.copy()