"""
image_ops.py — Opérations de traitement d'image réimplémentées avec NumPy.

Le sujet demande d'implémenter soi-même les opérations de base
(conversion couleur, seuillage, réduction du bruit, centroïde...).
Chaque fonction ici a un équivalent OpenCV (indiqué dans la docstring) :
le fichier tests/test_p1.py vérifie que nos résultats sont identiques
(ou quasi identiques) à ceux d'OpenCV.

Conventions (identiques à OpenCV, pour rester compatibles avec P2) :
  - image couleur : tableau uint8 de forme (H, W, 3), ordre des canaux BGR
  - HSV : H dans [0, 179], S et V dans [0, 255]
  - masque binaire : tableau bool de forme (H, W)
  - coordonnées : x = colonne, y = ligne, origine en haut à gauche
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1) Réduction de résolution et flou (réduction du bruit)
# ---------------------------------------------------------------------------
def downscale(img, factor):
    """Réduit la résolution d'un facteur entier par moyenne de blocs.

    Chaque bloc factor x factor devient un seul pixel (moyenne).
    Équivalent : cv2.resize(img, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)
    Intérêt : 4x moins de pixels à traiter avec factor=2, et la moyenne
    élimine déjà une partie du bruit du capteur.
    """
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    img = img[:h2 * factor, :w2 * factor].astype(np.float32)
    if img.ndim == 3:
        img = img.reshape(h2, factor, w2, factor, img.shape[2]).mean(axis=(1, 3))
    else:
        img = img.reshape(h2, factor, w2, factor).mean(axis=(1, 3))
    return np.round(img).astype(np.uint8)


def box_blur(img, k=3):
    """Flou moyenneur k x k (k impair), séparable : lignes puis colonnes.

    Équivalent : cv2.blur(img, (k, k)) (bords répliqués).
    """
    if k <= 1:
        return img
    r = k // 2
    f = img.astype(np.float32)
    pad = [(r, r), (0, 0)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(f, pad, mode="edge")
    f = sum(p[i:i + img.shape[0]] for i in range(k)) / k          # vertical
    pad = [(0, 0), (r, r)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(f, pad, mode="edge")
    f = sum(p[:, i:i + img.shape[1]] for i in range(k)) / k       # horizontal
    return np.round(f).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) Conversion de couleur BGR -> HSV
# ---------------------------------------------------------------------------
def bgr_to_hsv(img):
    """Convertit une image BGR uint8 en HSV (convention OpenCV).

    Équivalent : cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    Formules (par pixel) :
        V = max(R, G, B)
        S = (V - min) / V * 255            (0 si V = 0)
        H = 60*(G-B)/delta        si V = R
            120 + 60*(B-R)/delta  si V = G
            240 + 60*(R-G)/delta  si V = B      (puis +360 si négatif)
        H est divisé par 2 pour tenir dans un uint8 (0..179).
    Pourquoi HSV ? La teinte H dépend peu de l'éclairage : un marqueur
    vert reste « vert » (H ~ 60) qu'il soit à l'ombre ou en pleine lumière,
    alors qu'en BGR ses trois valeurs changent toutes.
    """
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = v - mn

    s = np.where(v > 0, delta / np.maximum(v, 1e-6) * 255.0, 0.0)

    d = np.where(delta > 0, delta, 1.0)            # évite la division par 0
    h = np.where(v == r, 60.0 * (g - b) / d,
        np.where(v == g, 120.0 + 60.0 * (b - r) / d,
                         240.0 + 60.0 * (r - g) / d))
    h = np.where(delta > 0, h, 0.0)
    h = np.where(h < 0, h + 360.0, h)
    h = np.round(h / 2.0)
    h = np.where(h >= 180, h - 180, h)

    return np.stack([h, np.round(s), v], axis=-1).astype(np.uint8)


def bgr_to_ycrcb(img):
    """Convertit une image BGR uint8 en YCrCb (convention OpenCV, BT.601).

    Équivalent : cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Formules (par pixel, delta = 128 pour uint8) :
        Y  = 0.299*R + 0.587*G + 0.114*B
        Cr = (R - Y)*0.713 + 128
        Cb = (B - Y)*0.564 + 128
    Pourquoi YCrCb ? La chrominance (Cr, Cb) isole efficacement la peau
    humaine indépendamment de la luminance Y (ombres / projecteur).
    """
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    ycrcb = np.stack([y, cr, cb], axis=-1)
    return np.clip(np.round(ycrcb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3) Seuillage (segmentation)
# ---------------------------------------------------------------------------
def in_range_hsv(hsv, lower, upper):
    """Seuillage HSV : renvoie un masque bool (True = pixel de la couleur cherchée).

    Équivalent : cv2.inRange(hsv, lower, upper) > 0
    Particularité : si lower[0] > upper[0], la plage de teinte « fait le tour »
    (utile pour le rouge : H dans [170..179] U [0..10]).
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hl, sl, vl = lower
    hu, su, vu = upper
    if hl <= hu:
        mask_h = (h >= hl) & (h <= hu)
    else:                                           # plage circulaire (rouge)
        mask_h = (h >= hl) | (h <= hu)
    return mask_h & (s >= sl) & (s <= su) & (v >= vl) & (v <= vu)


def skin_mask(hsv, ycrcb, cr_range=(133, 173), cb_range=(77, 127),
              s_range=(20, 200), v_min=45):
    """Segmentation multi-espace de la peau humaine (YCrCb + HSV).

    Combine la chrominance YCrCb (robuste aux variations d'éclairage) avec
    des garde-fous HSV (teinte peau, saturation non nulle, luminance min)
    pour rejeter les ombres sombres et les reflets blancs du tableau.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    ycrcb_ok = (cr >= cr_range[0]) & (cr <= cr_range[1]) & (cb >= cb_range[0]) & (cb <= cb_range[1])
    hue_ok = (h <= 25) | (h >= 165)
    hsv_ok = hue_ok & (s >= s_range[0]) & (s <= s_range[1]) & (v >= v_min)
    return ycrcb_ok & hsv_ok


def dark_achromatic_mask(hsv, v_max=80, s_max=95):
    """Isole les pixels sombres et peu saturés (télécommande noire).

    Un objet noir a une luminance V faible et une saturation S faible
    (sa teinte H est du bruit pur et ne doit pas être utilisée).
    """
    return (hsv[..., 2] <= v_max) & (hsv[..., 1] <= s_max)


def foreground_mask(frame_bgr, bg_model, thresh=22):
    """Soustraction de fond : pixels différant du modèle de fond statique.

    Utilise la distance L_inf (maximum sur les 3 canaux BGR) par rapport au
    modèle de fond adaptatif float32.
    """
    diff = np.abs(frame_bgr.astype(np.float32) - bg_model)
    return np.max(diff, axis=-1) > thresh


# ---------------------------------------------------------------------------
# 4) Morphologie mathématique (nettoyage du masque)
# ---------------------------------------------------------------------------
def _shift_reduce(mask, k, axis, op, pad_value):
    """Applique op (AND / OR) entre le masque et ses décalages de -r à +r."""
    r = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(mask, pad, mode="constant", constant_values=pad_value)
    n = mask.shape[axis]
    out = None
    for i in range(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        part = p[tuple(sl)]
        out = part.copy() if out is None else op(out, part)
    return out


def erode(mask, k=3):
    """Érosion binaire avec un élément structurant carré k x k.

    Un pixel reste True seulement si TOUT son voisinage k x k est True.
    -> supprime les petits points isolés (bruit) et amincit les objets.
    Équivalent : cv2.erode(mask, np.ones((k, k), np.uint8))
    Implémentation séparable : un carré = une ligne puis une colonne.
    """
    if k <= 1:
        return mask
    m = _shift_reduce(mask, k, 1, np.logical_and, True)
    return _shift_reduce(m, k, 0, np.logical_and, True)


def dilate(mask, k=3):
    """Dilatation binaire : un pixel devient True si UN voisin est True.

    -> bouche les petits trous et épaissit les objets.
    Équivalent : cv2.dilate(mask, np.ones((k, k), np.uint8))
    """
    if k <= 1:
        return mask
    m = _shift_reduce(mask, k, 1, np.logical_or, False)
    return _shift_reduce(m, k, 0, np.logical_or, False)


def opening(mask, k=3):
    """Ouverture = érosion puis dilatation : enlève le bruit (petits points)."""
    return dilate(erode(mask, k), k)


def closing(mask, k=3):
    """Fermeture = dilatation puis érosion : bouche les trous dans l'objet."""
    return erode(dilate(mask, k), k)


# ---------------------------------------------------------------------------
# 5) Composantes connexes (séparer les différentes « taches »)
# ---------------------------------------------------------------------------
def connected_components(mask, max_iter=1000):
    """Étiquetage des composantes connexes (8-connexité), en NumPy pur.

    Équivalent : cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    Principe (propagation du minimum) :
      1. chaque pixel blanc reçoit une étiquette unique (son indice + 1) ;
      2. à chaque itération, chaque pixel prend le minimum des étiquettes
         de ses 8 voisins blancs ;
      3. on répète jusqu'à ce que plus rien ne change : tous les pixels
         d'une même tache portent alors la même étiquette.
    Renvoie (labels, n) : labels = tableau int32 (0 = fond), n = nb de taches.
    """
    h, w = mask.shape
    big = np.int32(h * w + 1)
    labels = np.where(mask, np.arange(1, h * w + 1, dtype=np.int32).reshape(h, w), big)

    for _ in range(max_iter):
        p = np.pad(labels, 1, mode="constant", constant_values=big)
        m = labels
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                m = np.minimum(m, p[dy:dy + h, dx:dx + w])
        new = np.where(mask, m, big)
        # « saut de pointeur » : accélère beaucoup la convergence
        flat = new.ravel()
        fg = flat < big
        flat[fg] = np.minimum(flat[fg], flat[flat[fg] - 1])
        new = flat.reshape(h, w)
        if np.array_equal(new, labels):
            break
        labels = new

    # renumérotation 1..n
    out = np.zeros((h, w), dtype=np.int32)
    if mask.any():
        uniq, inv = np.unique(labels[mask], return_inverse=True)
        out[mask] = inv.astype(np.int32) + 1
        return out, len(uniq)
    return out, 0


def component_stats(labels, n):
    """Pour chaque tache : aire, centroïde (cx, cy) et boîte englobante.

    Le centroïde est la moyenne des coordonnées des pixels de la tache
    (moments d'ordre 0 et 1) :
        aire = m00 = nombre de pixels
        cx = m10 / m00 = moyenne des x      cy = m01 / m00 = moyenne des y
    Renvoie une liste de dicts triée par aire décroissante.
    """
    if n == 0:
        return []
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    area = np.bincount(lab, minlength=n + 1)
    sx = np.bincount(lab, weights=xs, minlength=n + 1)
    sy = np.bincount(lab, weights=ys, minlength=n + 1)
    xmin = np.full(n + 1, np.iinfo(np.int32).max); np.minimum.at(xmin, lab, xs)
    ymin = np.full(n + 1, np.iinfo(np.int32).max); np.minimum.at(ymin, lab, ys)
    xmax = np.full(n + 1, -1);                      np.maximum.at(xmax, lab, xs)
    ymax = np.full(n + 1, -1);                      np.maximum.at(ymax, lab, ys)
    stats = []
    for i in range(1, n + 1):
        if area[i] == 0:
            continue
        stats.append({
            "label": i,
            "area": int(area[i]),
            "cx": sx[i] / area[i],
            "cy": sy[i] / area[i],
            "bbox": (int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])),  # x0,y0,x1,y1
        })
    stats.sort(key=lambda s: s["area"], reverse=True)
    return stats
