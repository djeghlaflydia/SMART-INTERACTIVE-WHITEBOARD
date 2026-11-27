"""
remote.py — Module P3 : Détection des événements de la télécommande USB HID.

La télécommande de présentation sans fil (dongle USB 2.4 GHz) est reconnue
par le système d'exploitation comme un clavier standard (USB HID).

Mapping imposé par le cahier des charges :
  - Bouton NEXT (Suivant)     -> LEFT_CLICK  (CLIC_GAUCHE)
  - Bouton PREVIOUS (Précédent) -> RIGHT_CLICK (CLIC_DROIT)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

# Constantes d'actions standardisées (anglais pour P4/P5 + alias français P3)
LEFT_CLICK = "LEFT_CLICK"
RIGHT_CLICK = "RIGHT_CLICK"
NONE = None

CLIC_GAUCHE = LEFT_CLICK
CLIC_DROIT = RIGHT_CLICK
AUCUN_CLIC = NONE

# Codes étendus OpenCV (cv2.waitKeyEx) selon l'OS
# Bouton Suivant (Next) : Page Down / Flèche Droite / Flèche Bas
OPENCV_CODES_NEXT: Set[int] = {
    63277,    # macOS (Page Down)
    63235,    # macOS (Flèche Droite)
    63233,    # macOS (Flèche Bas)
    65366,    # Linux X11/GTK (Page Down)
    65363,    # Linux X11/GTK (Flèche Droite)
    65364,    # Linux X11/GTK (Flèche Bas)
    2228224,  # Windows (Page Down)
    2555904,  # Windows (Flèche Droite)
    2621440,  # Windows (Flèche Bas)
}

# Bouton Précédent (Previous) : Page Up / Flèche Gauche / Flèche Haut
OPENCV_CODES_PREVIOUS: Set[int] = {
    63276,    # macOS (Page Up)
    63234,    # macOS (Flèche Gauche)
    63232,    # macOS (Flèche Haut)
    65365,    # Linux X11/GTK (Page Up)
    65361,    # Linux X11/GTK (Flèche Gauche)
    65362,    # Linux X11/GTK (Flèche Haut)
    2162688,  # Windows (Page Up)
    2424832,  # Windows (Flèche Gauche)
    2490368,  # Windows (Flèche Haut)
}

# Raccourcis clavier de secours pour les tests sans télécommande
FALLBACK_CODES_NEXT: Set[int] = {ord("n"), ord("N")}
FALLBACK_CODES_PREVIOUS: Set[int] = {ord("p"), ord("P")}

# Ensembles complets exportés pour compatibilité directe avec telecommande.py
CODES_NEXT: Set[int] = OPENCV_CODES_NEXT | FALLBACK_CODES_NEXT
CODES_PREVIOUS: Set[int] = OPENCV_CODES_PREVIOUS | FALLBACK_CODES_PREVIOUS

# Noms de touches symboliques (Tkinter / frameworks GUI)
KEYSYMS_NEXT: Set[str] = {
    "Next", "Page_Down", "Right", "Down", "pagedown", "right", "down"
}
KEYSYMS_PREVIOUS: Set[str] = {
    "Prior", "Previous", "Page_Up", "Left", "Up", "pageup", "left", "up"
}


@dataclass(frozen=True)
class RemoteEvent:
    """Événement structuré généré lors d'un appui valide sur la télécommande."""

    action: str               # LEFT_CLICK ou RIGHT_CLICK
    button: str               # "NEXT" ou "PREVIOUS"
    raw_code: object          # code OpenCV (int) ou keysym (str)
    source: str = "hid"       # "hid" ou "keyboard_fallback"
    timestamp: float = field(default_factory=time.time)


def detecter_action(code_touche: int) -> Optional[str]:
    """
    Fonction stateless compatible avec le module P3 d'origine.

    Paramètre :
        code_touche (int) : code retourné par cv2.waitKeyEx()

    Retourne :
        "LEFT_CLICK" (CLIC_GAUCHE), "RIGHT_CLICK" (CLIC_DROIT) ou None.
    """
    if code_touche in CODES_NEXT:
        return LEFT_CLICK
    if code_touche in CODES_PREVIOUS:
        return RIGHT_CLICK
    return NONE


class RemoteController:
    """
    Contrôleur P3 avec anti-rebond, support multi-OS et statistiques.

    Résilience déconnexion/reconnexion :
      La télécommande étant un périphérique USB HID géré par le noyau OS,
      le débranchement et le rebranchement du dongle USB sont transparents
      pour cv2.waitKeyEx() ou les événements clavier GUI, sans réinitialisation.
    """

    def __init__(
        self,
        debounce_ms: float = 180.0,
        enable_keyboard_fallback: bool = True,
    ):
        self.debounce_seconds = max(0.0, float(debounce_ms) / 1000.0)
        self.enable_keyboard_fallback = bool(enable_keyboard_fallback)

        self._last_trigger_by_button: Dict[str, float] = {}
        self.total_next: int = 0
        self.total_previous: int = 0
        self.ignored_debounce: int = 0
        self.last_event: Optional[RemoteEvent] = None

    @classmethod
    def from_config(cls, config_path: str = "config.json") -> "RemoteController":
        """Construit un RemoteController depuis la section 'remote' de config.json."""
        path = Path(config_path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("remote", {})
        return cls(
            debounce_ms=cfg.get("debounce_ms", 180.0),
            enable_keyboard_fallback=cfg.get("enable_keyboard_fallback", True),
        )

    def _classify_opencv_code(self, code: int) -> Optional[tuple]:
        if code == -1:
            return None
        if code in OPENCV_CODES_NEXT:
            return (LEFT_CLICK, "NEXT", "hid")
        if code in OPENCV_CODES_PREVIOUS:
            return (RIGHT_CLICK, "PREVIOUS", "hid")
        if self.enable_keyboard_fallback:
            if code in FALLBACK_CODES_NEXT:
                return (LEFT_CLICK, "NEXT", "keyboard_fallback")
            if code in FALLBACK_CODES_PREVIOUS:
                return (RIGHT_CLICK, "PREVIOUS", "keyboard_fallback")
        return None

    def _classify_keysym(self, keysym: str) -> Optional[tuple]:
        if not keysym:
            return None
        if keysym in KEYSYMS_NEXT:
            return (LEFT_CLICK, "NEXT", "hid")
        if keysym in KEYSYMS_PREVIOUS:
            return (RIGHT_CLICK, "PREVIOUS", "hid")
        if self.enable_keyboard_fallback:
            low = keysym.lower()
            if low == "n":
                return (LEFT_CLICK, "NEXT", "keyboard_fallback")
            if low == "p":
                return (RIGHT_CLICK, "PREVIOUS", "keyboard_fallback")
        return None

    def _emit_if_not_debounced(
        self,
        classified: Optional[tuple],
        raw_code: object,
        now: Optional[float] = None,
    ) -> Optional[RemoteEvent]:
        if classified is None:
            return None

        action, button, source = classified
        ts = time.time() if now is None else float(now)

        last_ts = self._last_trigger_by_button.get(button)
        if last_ts is not None and (ts - last_ts) < self.debounce_seconds:
            self.ignored_debounce += 1
            return None

        self._last_trigger_by_button[button] = ts
        if button == "NEXT":
            self.total_next += 1
        else:
            self.total_previous += 1

        event = RemoteEvent(
            action=action,
            button=button,
            raw_code=raw_code,
            source=source,
            timestamp=ts,
        )
        self.last_event = event
        return event

    def process_opencv_key(
        self,
        key_code: int,
        now: Optional[float] = None,
    ) -> Optional[RemoteEvent]:
        """Traite un code issu de cv2.waitKeyEx() et retourne RemoteEvent ou None."""
        classified = self._classify_opencv_code(int(key_code))
        return self._emit_if_not_debounced(classified, int(key_code), now=now)

    def process_keysym(
        self,
        keysym: str,
        now: Optional[float] = None,
    ) -> Optional[RemoteEvent]:
        """Traite un nom de touche symbolique (Tkinter, etc.) et retourne RemoteEvent ou None."""
        classified = self._classify_keysym(keysym)
        return self._emit_if_not_debounced(classified, keysym, now=now)

    def get_stats(self) -> Dict[str, int]:
        """Retourne les compteurs d'événements détectés et filtrés."""
        return {
            "total_next": self.total_next,
            "total_previous": self.total_previous,
            "ignored_debounce": self.ignored_debounce,
        }
