"""
ui_regions.py — Gestionnaire léger de zones d'interface (indépendant du framework GUI).

Permet à P4 (et P5) d'enregistrer des boutons, palettes ou zones de dessin
dans le repère du tableau (1280x720) et d'effectuer un hit-testing exact
lors des clics gauche / droit.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class UIZone:
    """Zone rectangulaire interactive dans le repère du tableau."""

    zone_id: str
    x: float
    y: float
    width: float
    height: float
    label: str = ""
    z_index: int = 0
    enabled: bool = True
    on_click: Optional[Callable[[object], None]] = None

    def contains(self, px: float, py: float) -> bool:
        """Retourne True si (px, py) se trouve à l'intérieur du rectangle."""
        if not self.enabled:
            return False
        return (self.x <= px < self.x + self.width) and (self.y <= py < self.y + self.height)


class UIZoneManager:
    """Registre de zones d'interface avec résolution par z_index."""

    def __init__(self):
        self._zones: List[UIZone] = []

    def add_zone(self, zone: UIZone) -> None:
        """Ajoute ou remplace une zone ayant le même zone_id."""
        self.remove_zone(zone.zone_id)
        self._zones.append(zone)
        self._zones.sort(key=lambda z: z.z_index, reverse=True)

    def remove_zone(self, zone_id: str) -> None:
        """Supprime une zone par son identifiant."""
        self._zones = [z for z in self._zones if z.zone_id != zone_id]

    def clear(self) -> None:
        """Supprime toutes les zones enregistrées."""
        self._zones.clear()

    def hit_test(self, px: float, py: float) -> Optional[UIZone]:
        """Retourne la zone active au premier plan contenant (px, py), ou None."""
        for zone in self._zones:
            if zone.contains(px, py):
                return zone
        return None

    def zones(self) -> List[UIZone]:
        """Retourne la liste des zones enregistrées (triées par z_index décroissant)."""
        return list(self._zones)
