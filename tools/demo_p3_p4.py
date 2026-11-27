"""
demo_p3_p4.py
Démonstration intégrée P1 + P2 + P3 + P4 sur un tableau blanc 1280x720.

Modes d'exécution :
  1. Mode réel (caméra P1 + calibration P2 + télécommande P3 + interaction P4) :
       python3 -m tools.demo_p3_p4
  2. Mode simulation sans caméra (souris = pointeur, touche 'l' = perte temporaire) :
       python3 -m tools.demo_p3_p4 --simulate

Contrôles :
  - Bouton NEXT (ou touche 'n')     : CLIC GAUCHE (activer bouton UI ou dessiner un point/trait)
  - Bouton PREVIOUS (ou touche 'p') : CLIC DROIT  (changer de couleur / effacer dernier repère)
  - Touche 'c'                      : Effacer le tableau
  - Touche 'l' (mode --simulate)    : Simuler une perte temporaire du pointeur
  - Touche 'q' ou ESC               : Quitter
"""

import argparse
import time

import cv2
import numpy as np

from p1.tracker import PointerTracker
from p2.calibration import BoardCalibration
from p2.transformer import BoardTransformer
from p3.remote import RemoteController
from p4 import ClickType, InteractionManager, PointerStatus, UIZone, UIZoneManager

BOARD_W = 1280
BOARD_H = 720
TOOLBAR_H = 70

PALETTE = [
    ("btn_blue", "Bleu", (230, 130, 30)),
    ("btn_green", "Vert", (60, 190, 75)),
    ("btn_orange", "Orange", (30, 140, 255)),
    ("btn_purple", "Violet", (200, 80, 180)),
]


class IdentityTransformer:
    """Transformateur identité utilisé uniquement en mode --simulate."""

    def __init__(self, width=BOARD_W, height=BOARD_H):
        self.w = width
        self.h = height

    def transform_clamped(self, x, y):
        inside = (0.0 <= x < self.w) and (0.0 <= y < self.h)
        cx = max(0.0, min(self.w - 1.0, float(x)))
        cy = max(0.0, min(self.h - 1.0, float(y)))
        return cx, cy, inside


def build_ui_zones(state):
    """Enregistre les boutons de la barre d'outils dans UIZoneManager (P4)."""
    zones = UIZoneManager()
    x_cursor = 20

    for idx, (zid, label, bgr) in enumerate(PALETTE):
        def make_color_cb(color_idx, color_name):
            def _cb(event):
                if event.click_type == ClickType.LEFT:
                    state["color_idx"] = color_idx
                    state["eraser"] = False
                    state["status_msg"] = f"Couleur selectionnee : {color_name}"
            return _cb

        zones.add_zone(
            UIZone(
                zone_id=zid,
                x=x_cursor,
                y=12,
                width=110,
                height=46,
                label=label,
                z_index=10,
                on_click=make_color_cb(idx, label),
            )
        )
        x_cursor += 125

    def on_eraser(event):
        if event.click_type == ClickType.LEFT:
            state["eraser"] = not state["eraser"]
            state["status_msg"] = "Mode Gomme" if state["eraser"] else "Mode Dessin"

    zones.add_zone(
        UIZone(
            zone_id="btn_eraser",
            x=x_cursor,
            y=12,
            width=120,
            height=46,
            label="Gomme",
            z_index=10,
            on_click=on_eraser,
        )
    )
    x_cursor += 135

    def on_clear(event):
        if event.click_type == ClickType.LEFT:
            state["canvas"][TOOLBAR_H:, :] = 250
            state["last_draw_pt"] = None
            state["status_msg"] = "Tableau efface !"

    zones.add_zone(
        UIZone(
            zone_id="btn_clear",
            x=x_cursor,
            y=12,
            width=120,
            height=46,
            label="Effacer",
            z_index=10,
            on_click=on_clear,
        )
    )

    def on_canvas_click(event):
        bx, by = int(event.x), int(event.y)
        if event.click_type == ClickType.LEFT:
            if state["eraser"]:
                cv2.circle(state["canvas"], (bx, by), 22, (250, 250, 250), -1)
                state["status_msg"] = f"Gomme appliquee en ({bx}, {by})"
            else:
                color = PALETTE[state["color_idx"]][2]
                prev = state["last_draw_pt"]
                if prev is not None and (abs(prev[0] - bx) + abs(prev[1] - by)) < 160:
                    cv2.line(state["canvas"], prev, (bx, by), color, 6)
                else:
                    cv2.circle(state["canvas"], (bx, by), 7, color, -1)
                state["last_draw_pt"] = (bx, by)
                state["status_msg"] = f"Clic Gauche (NEXT) en ({bx}, {by})"
        elif event.click_type == ClickType.RIGHT:
            state["color_idx"] = (state["color_idx"] + 1) % len(PALETTE)
            state["eraser"] = False
            state["last_draw_pt"] = None
            name = PALETTE[state["color_idx"]][1]
            state["status_msg"] = f"Clic Droit (PREV) -> Couleur: {name}"

    zones.add_zone(
        UIZone(
            zone_id="canvas_area",
            x=0,
            y=TOOLBAR_H,
            width=BOARD_W,
            height=BOARD_H - TOOLBAR_H,
            label="Canvas",
            z_index=1,
            on_click=on_canvas_click,
        )
    )
    return zones


def render_frame(state, snapshot, ripples):
    """Construit l'image 1280x720 avec barre d'outils, curseur et feedback clics."""
    frame = state["canvas"].copy()

    # Barre d'outils
    cv2.rectangle(frame, (0, 0), (BOARD_W, TOOLBAR_H), (40, 44, 52), -1)

    for idx, (zid, label, bgr) in enumerate(PALETTE):
        zone = next(z for z in state["zones"].zones() if z.zone_id == zid)
        x0, y0 = int(zone.x), int(zone.y)
        x1, y1 = int(zone.x + zone.width), int(zone.y + zone.height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), bgr, -1)
        if not state["eraser"] and state["color_idx"] == idx:
            cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (255, 255, 255), 3)
        cv2.putText(frame, label, (x0 + 14, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for zid, active in [("btn_eraser", state["eraser"]), ("btn_clear", False)]:
        zone = next(z for z in state["zones"].zones() if z.zone_id == zid)
        x0, y0 = int(zone.x), int(zone.y)
        x1, y1 = int(zone.x + zone.width), int(zone.y + zone.height)
        fill = (120, 130, 150) if active else (70, 75, 88)
        cv2.rectangle(frame, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 185, 195), 1)
        cv2.putText(frame, zone.label, (x0 + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # HUD statut en haut à droite
    hud_text = f"{snapshot.status} | {state['status_msg']}"
    cv2.putText(frame, hud_text[:58], (710, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1)

    # Effets visuels des clics récents
    now = time.time()
    ripples[:] = [r for r in ripples if now - r["t"] < 0.45]
    for r in ripples:
        dt = now - r["t"]
        radius = int(10 + dt * 70)
        col = (60, 210, 90) if r["type"] == ClickType.LEFT else (40, 160, 255)
        cv2.circle(frame, (int(r["x"]), int(r["y"])), radius, col, 2)

    # Curseur du pointeur
    pos = snapshot.position()
    if pos is not None:
        px, py = int(pos[0]), int(pos[1])
        if snapshot.status == PointerStatus.TRACKING:
            c_col = (0, 180, 0)
        elif snapshot.status == PointerStatus.HOLD:
            c_col = (0, 180, 255)
        else:
            c_col = (0, 120, 255)
        cv2.circle(frame, (px, py), 10, c_col, 2)
        cv2.circle(frame, (px, py), 3, c_col, -1)
    elif snapshot.last_valid_x is not None:
        px, py = int(snapshot.last_valid_x), int(snapshot.last_valid_y)
        cv2.drawMarker(frame, (px, py), (0, 0, 220), cv2.MARKER_CROSS, 18, 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Démo intégrée P1 + P2 + P3 + P4")
    parser.add_argument("--simulate", action="store_true", help="Simuler le pointeur avec la souris")
    args = parser.parse_args()

    canvas = np.full((BOARD_H, BOARD_W, 3), 250, dtype=np.uint8)
    state = {
        "canvas": canvas,
        "color_idx": 0,
        "eraser": False,
        "last_draw_pt": None,
        "status_msg": "Pret (NEXT=Clic Gauche, PREV=Clic Droit)",
    }
    zones = build_ui_zones(state)
    state["zones"] = zones
    ripples = []

    tracker = None
    if args.simulate:
        transformer = IdentityTransformer(BOARD_W, BOARD_H)
    else:
        tracker = PointerTracker("config.json")
        tracker.start()
        calibration = BoardCalibration()
        corners = calibration.select_corners(tracker)
        if corners is None:
            tracker.stop()
            print("Calibration annulee.")
            return
        transformer = BoardTransformer(BOARD_W, BOARD_H)
        transformer.calibrate(corners, calibration.image_size)

    remote = RemoteController.from_config("config.json")
    manager = InteractionManager.from_config(
        board_transformer=transformer,
        config_path="config.json",
        remote_controller=remote,
        ui_zones=zones,
    )
    manager.on_event(lambda ev: ripples.append({"x": ev.x, "y": ev.y, "type": ev.click_type, "t": time.time()}))

    win_name = "Smart Interactive Whiteboard — P3 + P4"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    sim_mouse = {"pos": (640.0, 360.0), "lost": False}
    if args.simulate:
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                sim_mouse["pos"] = (float(x), float(y))
        cv2.setMouseCallback(win_name, on_mouse)

    try:
        while True:
            key = cv2.waitKeyEx(15)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state["canvas"][TOOLBAR_H:, :] = 250
                state["last_draw_pt"] = None
                state["status_msg"] = "Tableau efface !"
                key = -1
            if args.simulate and key in (ord("l"), ord("L")):
                sim_mouse["lost"] = not sim_mouse["lost"]
                state["status_msg"] = "Pointeur SIMULE PERDU" if sim_mouse["lost"] else "Pointeur retabli"
                key = -1

            pointer_input = None if (args.simulate and sim_mouse["lost"]) else (
                sim_mouse["pos"] if args.simulate else tracker.get_pointer()
            )
            snapshot, _ = manager.step(pointer_input, key_code=key)

            display = render_frame(state, snapshot, ripples)
            cv2.imshow(win_name, display)
    finally:
        if tracker is not None:
            tracker.stop()
        cv2.destroyAllWindows()
        print("Statistiques P4 :", manager.get_stats())


if __name__ == "__main__":
    main()
