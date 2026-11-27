"""
p3 — Configuration et détection des événements de la télécommande USB.
"""

from .remote import (
    AUCUN_CLIC,
    CLIC_DROIT,
    CLIC_GAUCHE,
    CODES_NEXT,
    CODES_PREVIOUS,
    LEFT_CLICK,
    NONE,
    RIGHT_CLICK,
    RemoteController,
    RemoteEvent,
    detecter_action,
)

__all__ = [
    "RemoteController",
    "RemoteEvent",
    "LEFT_CLICK",
    "RIGHT_CLICK",
    "NONE",
    "CLIC_GAUCHE",
    "CLIC_DROIT",
    "AUCUN_CLIC",
    "CODES_NEXT",
    "CODES_PREVIOUS",
    "detecter_action",
]
