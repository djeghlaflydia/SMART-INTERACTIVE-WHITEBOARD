"""
events.py — Structures de données et contrats d'interface entre P4 et P5.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ClickType:
    """Types de clics déclenchés par P4 vers l'interface graphique P5."""

    LEFT = "LEFT_CLICK"
    RIGHT = "RIGHT_CLICK"


class PointerStatus:
    """États possibles du pointeur dans le repère du tableau."""

    TRACKING = "TRACKING"          # Pointeur détecté et à l'intérieur du tableau
    HOLD = "HOLD"                  # Perte brève gérée par P1 (position maintenue)
    LAST_VALID = "LAST_VALID"      # Perte récente : dernière position valide conservée (délai de grâce)
    OUT_OF_BOARD = "OUT_OF_BOARD"  # Pointeur détecté mais en dehors de la zone calibrée
    LOST = "LOST"                  # Pointeur perdu au-delà du délai de grâce (clic impossible)


@dataclass(frozen=True)
class PointerSnapshot:
    """
    Instantané de la position du pointeur dans le repère logique du tableau.
    Fourni à P5 pour afficher le curseur et le statut en temps réel.
    """

    valid: bool                          # True si une position cliquable existe
    detected: bool                       # True si le marqueur/main est réellement vu sur cette image
    inside_board: bool                   # True si la position brute transformée est dans le tableau
    x: Optional[float]                   # Coordonnée X tableau utilisable (ou None)
    y: Optional[float]                   # Coordonnée Y tableau utilisable (ou None)
    last_valid_x: Optional[float]        # Dernière coordonnée X valide connue
    last_valid_y: Optional[float]        # Dernière coordonnée Y valide connue
    camera_x: Optional[float] = None     # Coordonnée X caméra d'origine (traçabilité)
    camera_y: Optional[float] = None     # Coordonnée Y caméra d'origine (traçabilité)
    status: str = PointerStatus.LOST
    lost_duration: float = 0.0           # Durée écoulée depuis la dernière détection valide (s)
    timestamp: float = field(default_factory=time.time)

    def position(self) -> Optional[tuple]:
        """Retourne (x, y) si le pointeur est utilisable, sinon None."""
        if self.valid and self.x is not None and self.y is not None:
            return (self.x, self.y)
        return None


@dataclass(frozen=True)
class InteractionEvent:
    """
    Événement complet généré lorsqu'un bouton de la télécommande est pressé
    alors qu'une position valide du pointeur est disponible.
    """

    click_type: str                      # ClickType.LEFT ou ClickType.RIGHT
    x: float                             # Coordonnée X sur le tableau (0 .. board_width - 1)
    y: float                             # Coordonnée Y sur le tableau (0 .. board_height - 1)
    camera_x: Optional[float]            # Coordonnée X caméra correspondante
    camera_y: Optional[float]            # Coordonnée Y caméra correspondante
    pointer_status: str                  # TRACKING / HOLD / LAST_VALID / OUT_OF_BOARD
    inside_board: bool                   # True si dans les limites exactes du tableau
    target_id: Optional[str] = None      # Identifiant de la zone UI touchée (si enregistrée)
    button: str = ""                     # "NEXT" ou "PREVIOUS"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
