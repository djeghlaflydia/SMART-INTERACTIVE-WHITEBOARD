# P1 — Acquisition caméra et détection du pointeur

Module P1 du projet **Smart Interactive Whiteboard** (USTHB, 2026-2027).
Il fournit la **position du pointeur dans l'image caméra** aux modules P2 (calibration) et P4 (interaction).

## Installation

```bash
pip install numpy opencv-python
```
Python 3.9 ou plus récent.

## Structure

```
P1_detection/
├── config.json          <- réglages partagés (caméra, seuils HSV, stabilisation)
├── p1/
│   ├── __init__.py
│   ├── camera.py        <- acquisition (OpenCV VideoCapture)
│   ├── image_ops.py     <- traitements réimplémentés en NumPy (HSV, seuillage, morphologie, composantes, centroïde)
│   ├── detector.py      <- pipeline de détection sur une image
│   ├── stabilizer.py    <- lissage temporel, gestion des pertes, PointerState
│   └── tracker.py       <- PointerTracker : point d'entrée pour l'équipe (thread)
├── tools/
│   ├── check_camera.py  <- étape 1 : trouver la caméra, vérifier l'image et les FPS
│   ├── tune_hsv.py      <- étape 2 : régler les seuils du marqueur (clic + curseurs), sauver
│   └── demo_p1.py       <- étape 3 : démo + tests (stabilité, journal CSV, captures)
└── tests/test_p1.py     <- tests automatiques sans caméra (NumPy vs OpenCV, précision...)
```

## Ordre d'utilisation

1. `python tools/check_camera.py` puis mettre le bon `camera.index` dans `config.json`
2. `python tools/tune_hsv.py` : cliquer sur le marqueur, ajuster, touche `s`
3. `python tools/demo_p1.py` : vérifier le suivi ; touche `j` pour le test de stabilité
4. `python tests/test_p1.py` : tests automatiques

## Contrat d'interface (à partager avec P2, P4 et P5)

```python
from p1 import PointerTracker

tracker = PointerTracker("config.json")
tracker.start()                    # ouvre la caméra, lance le thread

state = tracker.get_pointer()      # PointerState (dernier état, non bloquant)
pos   = state.position()           # (x, y) float en PIXELS CAMÉRA, ou None
frame = tracker.get_frame()        # dernière image BGR (pour la calibration de P2)
w, h  = tracker.frame_size         # résolution réelle de la caméra

tracker.stop()
```

| Champ de `PointerState` | Type | Signification |
|---|---|---|
| `valid` | bool | `x, y` utilisables (marqueur vu, ou perdu depuis moins de 5 images) |
| `detected` | bool | marqueur réellement vu sur cette image |
| `status` | str | `"TRACKING"`, `"HOLD"` (perte brève, dernière position conservée) ou `"LOST"` |
| `x, y` | float | position **lissée**, à utiliser |
| `raw_x, raw_y` | float | position brute (diagnostic) |
| `area`, `bbox` | int, tuple | aire en pixels, boîte `(x0, y0, x1, y1)` |
| `frame_index`, `timestamp` | int, float | numéro d'image, horodatage |
| `frame_size` | (w, h) | résolution de l'image caméra |

**Conventions :** x = colonne, y = ligne, origine **en haut à gauche** de l'image caméra,
résolution réelle de la caméra (par défaut 640×480). Pas de miroir. Ce sont les mêmes
conventions qu'OpenCV : P2 peut appliquer son homographie directement sur `(x, y)`.

**Règles communes :**
- P2 calibre avec `tracker.get_frame()` → même caméra, même résolution que la détection.
- Après la calibration, la caméra ne bouge plus (sinon refaire la calibration).
- P5 : **ne jamais utiliser la couleur du marqueur dans l'interface projetée**
  (ni pour le pointeur affiché, ni pour les boutons ou le dessin), sinon la caméra
  détecte la projection au lieu du marqueur.
- Un seul programme ouvre la caméra : le `PointerTracker`. Les autres modules lisent ses résultats.


---

# P2 — Calibration et transformation homographique

Module P2 du projet **Smart Interactive Whiteboard** .

Il assure la calibration du tableau interactif et la transformation des coordonnées du pointeur depuis le repère de la caméra vers le repère logique du tableau.

## Fonctionnalités

- Sélection des quatre coins du tableau à partir de l'image caméra.
- Organisation automatique des coins dans le bon ordre.
- Validation géométrique des points sélectionnés.
- Calcul de la matrice d'homographie.
- Transformation des coordonnées caméra vers les coordonnées du tableau.
- Vérification si le pointeur se trouve à l'intérieur du tableau.
- Limitation des coordonnées aux dimensions du tableau.
- Vérification de l'erreur de reprojection.

## Structure

```text
p2/
├── __init__.py
├── calibration.py     <- sélection et validation des quatre coins
├── homography.py      <- calcul et application de l'homographie
└── transformer.py     <- interface principale pour les autres modules
```

## Repère du tableau

Le repère logique du tableau est défini par défaut avec une résolution de **1280 × 720 pixels**.

- Origine : coin supérieur gauche.
- Axe X : augmente vers la droite.
- Axe Y : augmente vers le bas.

## Contrat d'interface

Le module P2 utilise les coordonnées fournies par P1 et les transforme en coordonnées logiques du tableau.

```python
from p2.transformer import BoardTransformer

transformer = BoardTransformer(1280, 720)

# Calibration à partir des quatre coins détectés dans l'image caméra
transformer.calibrate(camera_points, image_size)

# Transformation d'une position caméra
board_x, board_y = transformer.transform(camera_x, camera_y)

# Vérification de la position
inside = transformer.is_inside(board_x, board_y)
```

**Entrées :** coordonnées du pointeur dans l'image caméra, fournies par P1.

**Sorties :** coordonnées transformées dans le repère logique du tableau, destinées aux modules P4 et P5.

## Tests et démonstration

Depuis la racine du projet :

```bash
python -m tests.test_p2
```

Pour lancer la démonstration en temps réel :

```bash
python -m tools.demo_p2
```

La démonstration utilise le `PointerTracker` de P1 pour récupérer les coordonnées du pointeur et visualiser leur transformation sur le tableau.


---

# P3 — Configuration et détection des événements de la télécommande

Module P3 du projet **Smart Interactive Whiteboard**.

La télécommande sans fil 2.4 GHz (dongle USB) est reconnue comme un périphérique standard **USB HID (clavier)**.

- **Bouton Next (Suivant)** → `LEFT_CLICK` (`CLIC_GAUCHE`)
- **Bouton Previous (Précédent)** → `RIGHT_CLICK` (`CLIC_DROIT`)

## Contrat d'interface

```python
from p3 import RemoteController, LEFT_CLICK, RIGHT_CLICK

remote = RemoteController.from_config("config.json")
event = remote.process_opencv_key(cv2.waitKeyEx(1))
if event is not None:
    print(event.action, event.button)  # LEFT_CLICK / RIGHT_CLICK
```

## Tests P3

```bash
python3 -m tests.test_p3
python3 -m tools.test_p3_remote
```

---

# P4 — Interaction et gestion des événements

Module P4 du projet **Smart Interactive Whiteboard**.

Relie en temps réel :
- **P1 (`PointerTracker`)** : position caméra du pointeur
- **P2 (`BoardTransformer`)** : transformation homographique vers le repère tableau `1280 × 720`
- **P3 (`RemoteController`)** : détection des clics gauche/droite

Fonctionnalités :
- Conservation de la dernière position valide du pointeur (`HOLD` + fenêtre de grâce configurable `lost_grace_seconds`).
- Rejet ou clamp configurable des clics hors tableau.
- Hit-testing optionnel sur les éléments d'interface (`UIZoneManager`).
- File d'événements thread-safe (`pop_events()`) et callbacks (`on_left_click`, `on_right_click`, `on_event`) pour **P5**.

## Contrat d'interface (à partager avec P5)

```python
from p4 import InteractionManager, ClickType

manager = InteractionManager.from_config(transformer, "config.json")

# Dans la boucle principale :
snapshot, event = manager.step(tracker.get_pointer(), key_code=cv2.waitKeyEx(1))

# Ou via file d'événements / callbacks :
for ev in manager.pop_events():
    if ev.click_type == ClickType.LEFT:
        print(f"Clic gauche en ({ev.x:.0f}, {ev.y:.0f}) cible={ev.target_id}")
    elif ev.click_type == ClickType.RIGHT:
        print(f"Clic droit en ({ev.x:.0f}, {ev.y:.0f})")
```

## Tests et démonstration P3 + P4

```bash
python3 -m tests.test_p4
python3 -m tools.demo_p3_p4 --simulate   # Sans caméra (souris + télécommande/clavier)
python3 -m tools.demo_p3_p4              # Complet (P1 + P2 + P3 + P4)
```


