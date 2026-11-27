"""
p4 — Interaction et gestion des événements du tableau blanc interactif.

Relie :
  - P1 (PointerState en coordonnées caméra)
  - P2 (BoardTransformer -> coordonnées logiques du tableau 1280x720)
  - P3 (RemoteController -> LEFT_CLICK / RIGHT_CLICK)
pour produire des InteractionEvent exploitables par P5 (interface graphique).
"""

from .events import (
    ClickType,
    InteractionEvent,
    PointerSnapshot,
    PointerStatus,
)
from .interaction import InteractionManager
from .ui_regions import UIZone, UIZoneManager

__all__ = [
    "ClickType",
    "PointerStatus",
    "PointerSnapshot",
    "InteractionEvent",
    "UIZone",
    "UIZoneManager",
    "InteractionManager",
]
