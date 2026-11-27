"""
test_p4.py
Tests automatisés de P4 (intégration P1 PointerState + P2 BoardTransformer + P3 RemoteController) :
  - Clic gauche (Next) et clic droit (Previous) déclenchés séparément à la position exacte
  - Suivi du pointeur en mouvement et cohérence caméra <-> tableau <-> action
  - Gestion de la perte temporaire du pointeur (HOLD + fenêtre de grâce LAST_VALID -> LOST)
  - Comportement hors tableau
  - Interaction avec des éléments d'interface (UIZoneManager + callbacks)

Lancer avec :
    python3 -m tests.test_p4
"""

import numpy as np

from p1.stabilizer import PointerState
from p2.transformer import BoardTransformer
from p3 import RemoteController
from p4 import ClickType, InteractionManager, PointerStatus, UIZone, UIZoneManager

W, H = 1280, 720

# Coins caméra synthétiques identiques à test_p2.py (perspective réaliste)
CAMERA_CORNERS = np.array([
    [210, 120],  # TL
    [520, 100],  # TR
    [560, 330],  # BR
    [180, 340],  # BL
], dtype=np.float32)


def make_calibrated_transformer() -> BoardTransformer:
    t = BoardTransformer(W, H)
    t.calibrate(CAMERA_CORNERS)
    return t


def make_pointer_state(x: float, y: float, status: str = "TRACKING", detected: bool = True) -> PointerState:
    return PointerState(
        valid=True,
        detected=detected,
        status=status,
        x=x,
        y=y,
        raw_x=x,
        raw_y=y,
        frame_size=(640, 480),
    )


def test_clicks_at_current_pointer_position():
    """Vérifie séparément le clic gauche et le clic droit à la position courante."""
    transformer = make_calibrated_transformer()
    remote = RemoteController(debounce_ms=0)
    manager = InteractionManager(transformer, remote_controller=remote)

    # Centre approximatif en caméra -> calculons la position tableau exacte attendue
    cam_x, cam_y = 365.0, 220.0
    expected_bx, expected_by = transformer.transform(cam_x, cam_y)

    # 1. Clic gauche (bouton NEXT)
    snap, ev_left = manager.step(make_pointer_state(cam_x, cam_y), key_code=63277, now=1.0)
    assert snap.valid and snap.status == PointerStatus.TRACKING
    assert ev_left is not None
    assert ev_left.click_type == ClickType.LEFT
    assert abs(ev_left.x - expected_bx) < 1e-6
    assert abs(ev_left.y - expected_by) < 1e-6
    assert ev_left.camera_x == cam_x and ev_left.camera_y == cam_y

    # 2. Clic droit (bouton PREVIOUS)
    _, ev_right = manager.step(make_pointer_state(cam_x, cam_y), key_code=63276, now=1.5)
    assert ev_right is not None
    assert ev_right.click_type == ClickType.RIGHT
    assert abs(ev_right.x - expected_bx) < 1e-6
    assert abs(ev_right.y - expected_by) < 1e-6
    print("OK  clic gauche et clic droit a la position exacte du pointeur")


def test_moving_pointer_coherence():
    """Vérifie le suivi lorsque le pointeur se déplace et la cohérence des coordonnées."""
    transformer = make_calibrated_transformer()
    manager = InteractionManager(transformer, remote_controller=RemoteController(debounce_ms=0))

    trajectory = [(230.0, 140.0), (350.0, 210.0), (490.0, 290.0)]
    triggered_xs = []

    for idx, (cx, cy) in enumerate(trajectory):
        expected_bx, expected_by = transformer.transform(cx, cy)
        snap, ev = manager.step(make_pointer_state(cx, cy), key_code=63277, now=2.0 + idx)
        assert ev is not None
        assert abs(ev.x - expected_bx) < 1e-6
        assert abs(ev.y - expected_by) < 1e-6
        triggered_xs.append(ev.x)

    assert triggered_xs[0] < triggered_xs[1] < triggered_xs[2]
    assert len(manager.pop_events()) == 3
    assert len(manager.pop_events()) == 0
    print("OK  coherence lorsque le pointeur se deplace")


def test_pointer_loss_hold_grace_and_expired():
    """Vérifie HOLD, conservation de la dernière position valide (grâce), puis LOST."""
    transformer = make_calibrated_transformer()
    manager = InteractionManager(
        transformer,
        remote_controller=RemoteController(debounce_ms=0),
        lost_grace_seconds=0.5,
    )

    cam_x, cam_y = 365.0, 220.0
    expected_bx, expected_by = transformer.transform(cam_x, cam_y)

    # Pointeur vu à t=10.0
    snap = manager.update_pointer(make_pointer_state(cam_x, cam_y), now=10.0)
    assert snap.status == PointerStatus.TRACKING

    # État HOLD de P1 à t=10.1 -> clic autorisé
    _, ev_hold = manager.step(
        make_pointer_state(cam_x, cam_y, status="HOLD", detected=False),
        key_code=63277,
        now=10.1,
    )
    assert ev_hold is not None and ev_hold.pointer_status == PointerStatus.HOLD

    # Perte totale P1 (None) à t=10.3 (dans les 0.5s de grâce) -> dernière position valide utilisée
    snap_grace, ev_grace = manager.step(None, key_code=63276, now=10.3)
    assert snap_grace.valid and snap_grace.status == PointerStatus.LAST_VALID
    assert ev_grace is not None
    assert ev_grace.click_type == ClickType.RIGHT
    assert abs(ev_grace.x - expected_bx) < 1e-6
    assert abs(ev_grace.y - expected_by) < 1e-6

    # Au-delà de 0.5s (t=10.9) -> pointeur LOST, clic rejeté
    snap_lost, ev_lost = manager.step(None, key_code=63277, now=10.9)
    assert not snap_lost.valid and snap_lost.status == PointerStatus.LOST
    assert ev_lost is None
    assert manager.get_stats()["dropped_pointer_lost"] == 1
    print("OK  gestion perte pointeur (HOLD -> LAST_VALID -> LOST)")


def test_out_of_board_rejection():
    """Vérifie que les clics hors de la zone calibrée sont rejetés si clamp_outside_clicks=False."""
    transformer = make_calibrated_transformer()
    manager = InteractionManager(
        transformer,
        remote_controller=RemoteController(debounce_ms=0),
        clamp_outside_clicks=False,
    )

    # Point caméra clairement en dehors du quadrilatère
    snap, ev = manager.step(make_pointer_state(50.0, 50.0), key_code=63277, now=1.0)
    assert not snap.valid
    assert snap.status == PointerStatus.OUT_OF_BOARD
    assert ev is None
    assert manager.get_stats()["dropped_out_of_board"] == 1
    print("OK  rejet des clics hors tableau")


def test_ui_zones_and_callbacks():
    """Vérifie les interactions avec différents éléments de l'interface (UIZoneManager)."""
    transformer = make_calibrated_transformer()
    zones = UIZoneManager()
    clicked_log = []

    zones.add_zone(UIZone(
        zone_id="canvas",
        x=0,
        y=0,
        width=W,
        height=H,
        z_index=1,
        on_click=lambda ev: clicked_log.append(("canvas", ev.click_type)),
    ))
    zones.add_zone(UIZone(
        zone_id="btn_save",
        x=500,
        y=250,
        width=280,
        height=220,
        z_index=10,
        on_click=lambda ev: clicked_log.append(("btn_save", ev.click_type)),
    ))

    manager = InteractionManager(
        transformer,
        remote_controller=RemoteController(debounce_ms=0),
        ui_zones=zones,
    )

    # Centre du tableau (~640, 360) -> doit toucher btn_save (z_index=10 > canvas)
    inv_h = np.linalg.inv(transformer.get_matrix())
    vec = inv_h @ np.array([640.0, 360.0, 1.0])
    cam_center = (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    _, ev_btn = manager.step(make_pointer_state(*cam_center), key_code=63277, now=1.0)
    assert ev_btn is not None and ev_btn.target_id == "btn_save"

    # Coin haut gauche tableau (~50, 50) -> doit toucher canvas
    vec2 = inv_h @ np.array([50.0, 50.0, 1.0])
    cam_corner = (float(vec2[0] / vec2[2]), float(vec2[1] / vec2[2]))
    _, ev_canvas = manager.step(make_pointer_state(*cam_corner), key_code=63276, now=2.0)
    assert ev_canvas is not None and ev_canvas.target_id == "canvas"

    assert clicked_log == [
        ("btn_save", ClickType.LEFT),
        ("canvas", ClickType.RIGHT),
    ]
    print("OK  interactions avec les elements UI (hit-testing + callbacks)")


def main():
    test_clicks_at_current_pointer_position()
    test_moving_pointer_coherence()
    test_pointer_loss_hold_grace_and_expired()
    test_out_of_board_rejection()
    test_ui_zones_and_callbacks()
    print("\nTous les tests P4 sont passes.")


if __name__ == "__main__":
    main()
