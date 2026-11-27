"""
interaction.py — Module P4 : Interaction et gestion des événements.

Responsabilités :
  1. Récupérer la position du pointeur (P1) et la transformer vers le repère
     du tableau via l'homographie (P2).
  2. Conserver la dernière position valide du pointeur et gérer les pertes
     momentanées (état HOLD de P1 + fenêtre de grâce configurable).
  3. Récupérer les événements de la télécommande (P3) et déclencher :
       - Clic gauche (LEFT_CLICK)  pour le bouton NEXT
       - Clic droit  (RIGHT_CLICK) pour le bouton PREVIOUS
     exactement à la position du pointeur sur le tableau.
  4. Transmettre les actions à l'interface graphique (P5) via callbacks,
     hit-testing des zones UI et file d'événements thread-safe.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

from p3.remote import LEFT_CLICK, RIGHT_CLICK, RemoteController, RemoteEvent

from .events import (
    ClickType,
    InteractionEvent,
    PointerSnapshot,
    PointerStatus,
)
from .ui_regions import UIZoneManager


class InteractionManager:
    """
    Orchestrateur P4 reliant P1 (pointeur), P2 (homographie), P3 (télécommande) et P5 (GUI).
    """

    def __init__(
        self,
        board_transformer,
        remote_controller: Optional[RemoteController] = None,
        lost_grace_seconds: float = 0.6,
        clamp_outside_clicks: bool = False,
        max_event_history: int = 256,
        ui_zones: Optional[UIZoneManager] = None,
    ):
        self.transformer = board_transformer
        self.remote = remote_controller or RemoteController()
        self.lost_grace_seconds = max(0.0, float(lost_grace_seconds))
        self.clamp_outside_clicks = bool(clamp_outside_clicks)
        self.ui_zones = ui_zones or UIZoneManager()

        self._lock = threading.Lock()

        # État mémorisé du pointeur
        self._last_valid_board: Optional[Tuple[float, float]] = None
        self._last_valid_camera: Optional[Tuple[float, float]] = None
        self._last_valid_time: Optional[float] = None
        self._snapshot = PointerSnapshot(
            valid=False,
            detected=False,
            inside_board=False,
            x=None,
            y=None,
            last_valid_x=None,
            last_valid_y=None,
            status=PointerStatus.LOST,
        )

        # File d'événements pour consommation asynchrone par P5
        self._event_queue: Deque[InteractionEvent] = deque(maxlen=max_event_history)
        self._event_history: Deque[InteractionEvent] = deque(maxlen=max_event_history)

        # Callbacks optionnels vers P5
        self._on_left_click: List[Callable[[InteractionEvent], None]] = []
        self._on_right_click: List[Callable[[InteractionEvent], None]] = []
        self._on_event: List[Callable[[InteractionEvent], None]] = []
        self._on_pointer_move: List[Callable[[PointerSnapshot], None]] = []

        # Statistiques de cohérence et diagnostic
        self.stats: Dict[str, int] = {
            "left_clicks": 0,
            "right_clicks": 0,
            "clicks_during_hold": 0,
            "clicks_during_grace": 0,
            "dropped_pointer_lost": 0,
            "dropped_out_of_board": 0,
        }

    @classmethod
    def from_config(
        cls,
        board_transformer,
        config_path: str = "config.json",
        remote_controller: Optional[RemoteController] = None,
        ui_zones: Optional[UIZoneManager] = None,
    ) -> "InteractionManager":
        """Construit un InteractionManager depuis config.json."""
        path = Path(config_path)
        cfg = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f).get("interaction", {})

        remote = remote_controller or RemoteController.from_config(config_path)
        return cls(
            board_transformer=board_transformer,
            remote_controller=remote,
            lost_grace_seconds=cfg.get("lost_grace_seconds", 0.6),
            clamp_outside_clicks=cfg.get("clamp_outside_clicks", False),
            ui_zones=ui_zones,
        )

    # ------------------------------------------------------------------
    # Abonnements callbacks pour P5
    # ------------------------------------------------------------------

    def on_left_click(self, callback: Callable[[InteractionEvent], None]) -> None:
        self._on_left_click.append(callback)

    def on_right_click(self, callback: Callable[[InteractionEvent], None]) -> None:
        self._on_right_click.append(callback)

    def on_event(self, callback: Callable[[InteractionEvent], None]) -> None:
        self._on_event.append(callback)

    def on_pointer_move(self, callback: Callable[[PointerSnapshot], None]) -> None:
        self._on_pointer_move.append(callback)

    # ------------------------------------------------------------------
    # Mise à jour de la position du pointeur (P1 -> P2 -> P4)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pointer_fields(
        pointer_state: Union[object, Tuple[float, float], None],
    ) -> Tuple[Optional[Tuple[float, float]], bool, str]:
        """Extrait (position_camera, detected, p1_status) depuis PointerState ou tuple."""
        if pointer_state is None:
            return None, False, "LOST"

        if isinstance(pointer_state, tuple) and len(pointer_state) == 2:
            return (float(pointer_state[0]), float(pointer_state[1])), True, "TRACKING"

        pos = pointer_state.position() if hasattr(pointer_state, "position") else None
        detected = bool(getattr(pointer_state, "detected", pos is not None))
        status = str(getattr(pointer_state, "status", "TRACKING" if pos else "LOST"))
        return pos, detected, status

    def _build_lost_snapshot(self, now: float) -> PointerSnapshot:
        """Construit le snapshot lorsque P1 ne fournit plus de position valide."""
        if self._last_valid_board is not None and self._last_valid_time is not None:
            lost_duration = max(0.0, now - self._last_valid_time)
            lx, ly = self._last_valid_board
            cx, cy = self._last_valid_camera if self._last_valid_camera else (None, None)

            if lost_duration <= self.lost_grace_seconds:
                return PointerSnapshot(
                    valid=True,
                    detected=False,
                    inside_board=True,
                    x=lx,
                    y=ly,
                    last_valid_x=lx,
                    last_valid_y=ly,
                    camera_x=cx,
                    camera_y=cy,
                    status=PointerStatus.LAST_VALID,
                    lost_duration=lost_duration,
                    timestamp=now,
                )

            return PointerSnapshot(
                valid=False,
                detected=False,
                inside_board=False,
                x=None,
                y=None,
                last_valid_x=lx,
                last_valid_y=ly,
                camera_x=cx,
                camera_y=cy,
                status=PointerStatus.LOST,
                lost_duration=lost_duration,
                timestamp=now,
            )

        return PointerSnapshot(
            valid=False,
            detected=False,
            inside_board=False,
            x=None,
            y=None,
            last_valid_x=None,
            last_valid_y=None,
            status=PointerStatus.LOST,
            lost_duration=0.0,
            timestamp=now,
        )

    def update_pointer(
        self,
        pointer_state: Union[object, Tuple[float, float], None],
        now: Optional[float] = None,
    ) -> PointerSnapshot:
        """
        Met à jour l'état du pointeur dans le repère du tableau.
        Gère la transformation homographique P2 et la perte temporaire.
        """
        ts = time.time() if now is None else float(now)
        cam_pos, detected, p1_status = self._extract_pointer_fields(pointer_state)

        with self._lock:
            if cam_pos is None:
                snapshot = self._build_lost_snapshot(ts)
            else:
                cam_x, cam_y = float(cam_pos[0]), float(cam_pos[1])
                try:
                    board_x, board_y, inside = self.transformer.transform_clamped(cam_x, cam_y)
                except (ValueError, RuntimeError):
                    snapshot = self._build_lost_snapshot(ts)
                else:
                    usable = inside or self.clamp_outside_clicks
                    if usable:
                        self._last_valid_board = (float(board_x), float(board_y))
                        self._last_valid_camera = (cam_x, cam_y)
                        self._last_valid_time = ts

                        if not inside:
                            status = PointerStatus.OUT_OF_BOARD
                        elif p1_status == "HOLD" or not detected:
                            status = PointerStatus.HOLD
                        else:
                            status = PointerStatus.TRACKING

                        snapshot = PointerSnapshot(
                            valid=True,
                            detected=detected,
                            inside_board=inside,
                            x=float(board_x),
                            y=float(board_y),
                            last_valid_x=float(board_x),
                            last_valid_y=float(board_y),
                            camera_x=cam_x,
                            camera_y=cam_y,
                            status=status,
                            lost_duration=0.0,
                            timestamp=ts,
                        )
                    else:
                        # Hors du tableau et clamp_outside_clicks=False
                        lx, ly = self._last_valid_board if self._last_valid_board else (None, None)
                        snapshot = PointerSnapshot(
                            valid=False,
                            detected=detected,
                            inside_board=False,
                            x=None,
                            y=None,
                            last_valid_x=lx,
                            last_valid_y=ly,
                            camera_x=cam_x,
                            camera_y=cam_y,
                            status=PointerStatus.OUT_OF_BOARD,
                            lost_duration=0.0,
                            timestamp=ts,
                        )

            self._snapshot = snapshot

        for cb in self._on_pointer_move:
            cb(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # Traitement des événements télécommande (P3 -> P4 -> P5)
    # ------------------------------------------------------------------

    def handle_remote_key(
        self,
        key_code: int,
        now: Optional[float] = None,
    ) -> Optional[InteractionEvent]:
        """Lit un code OpenCV via P3 et déclenche l'interaction correspondante."""
        remote_event = self.remote.process_opencv_key(key_code, now=now)
        if remote_event is None:
            return None
        return self.trigger_remote_event(remote_event)

    def handle_remote_keysym(
        self,
        keysym: str,
        now: Optional[float] = None,
    ) -> Optional[InteractionEvent]:
        """Lit un keysym GUI via P3 et déclenche l'interaction correspondante."""
        remote_event = self.remote.process_keysym(keysym, now=now)
        if remote_event is None:
            return None
        return self.trigger_remote_event(remote_event)

    def trigger_remote_event(self, remote_event: RemoteEvent) -> Optional[InteractionEvent]:
        """
        Associe un RemoteEvent (P3) à la position courante/valide du pointeur
        et émet un InteractionEvent vers P5.
        """
        with self._lock:
            snap = self._snapshot

            # Si le pointeur est en perte récente, recalculer la durée écoulée
            if snap.status == PointerStatus.LAST_VALID and self._last_valid_time is not None:
                elapsed = max(0.0, remote_event.timestamp - self._last_valid_time)
                if elapsed > self.lost_grace_seconds:
                    snap = self._build_lost_snapshot(remote_event.timestamp)
                    self._snapshot = snap

            if not snap.valid or snap.x is None or snap.y is None:
                if snap.status == PointerStatus.OUT_OF_BOARD:
                    self.stats["dropped_out_of_board"] += 1
                else:
                    self.stats["dropped_pointer_lost"] += 1
                return None

            click_type = ClickType.LEFT if remote_event.action == LEFT_CLICK else ClickType.RIGHT
            zone = self.ui_zones.hit_test(snap.x, snap.y)
            target_id = zone.zone_id if zone is not None else None

            event = InteractionEvent(
                click_type=click_type,
                x=snap.x,
                y=snap.y,
                camera_x=snap.camera_x,
                camera_y=snap.camera_y,
                pointer_status=snap.status,
                inside_board=snap.inside_board,
                target_id=target_id,
                button=remote_event.button,
                timestamp=remote_event.timestamp,
                metadata={
                    "raw_code": remote_event.raw_code,
                    "source": remote_event.source,
                    "lost_duration": snap.lost_duration,
                },
            )

            if click_type == ClickType.LEFT:
                self.stats["left_clicks"] += 1
            else:
                self.stats["right_clicks"] += 1

            if snap.status == PointerStatus.HOLD:
                self.stats["clicks_during_hold"] += 1
            elif snap.status == PointerStatus.LAST_VALID:
                self.stats["clicks_during_grace"] += 1

            self._event_queue.append(event)
            self._event_history.append(event)

        # Exécution des callbacks hors verrou
        if zone is not None and zone.on_click is not None:
            zone.on_click(event)

        if click_type == ClickType.LEFT:
            for cb in self._on_left_click:
                cb(event)
        else:
            for cb in self._on_right_click:
                cb(event)

        for cb in self._on_event:
            cb(event)

        return event

    def step(
        self,
        pointer_state: Union[object, Tuple[float, float], None],
        key_code: int = -1,
        now: Optional[float] = None,
    ) -> Tuple[PointerSnapshot, Optional[InteractionEvent]]:
        """
        Point d'entrée unique par itération de boucle principale :
        met à jour le pointeur puis traite un éventuel appui télécommande.
        """
        snapshot = self.update_pointer(pointer_state, now=now)
        event = self.handle_remote_key(key_code, now=now) if key_code != -1 else None
        return snapshot, event

    # ------------------------------------------------------------------
    # Accesseurs P5 & Diagnostics
    # ------------------------------------------------------------------

    def get_pointer_snapshot(self) -> PointerSnapshot:
        """Retourne le dernier PointerSnapshot calculé."""
        with self._lock:
            return self._snapshot

    def pop_events(self) -> List[InteractionEvent]:
        """Récupère et vide la file des événements en attente pour P5."""
        with self._lock:
            events = list(self._event_queue)
            self._event_queue.clear()
            return events

    def get_stats(self) -> Dict[str, int]:
        """Retourne une copie des statistiques d'interaction."""
        with self._lock:
            return dict(self.stats)
