"""
test_p3.py
Tests automatisés de P3 (sans matériel) :
  - Détectabilité et distinction Next -> LEFT_CLICK / Previous -> RIGHT_CLICK
  - Absence de déclenchement sans appui (-1 ou touches non mappées)
  - Support multi-OS (macOS, Linux, Windows) et keysyms
  - Appuis successifs et filtrage anti-rebond (debounce)
  - Compatibilité avec detecter_action()

Lancer avec :
    python3 -m tests.test_p3
"""

from p3 import (
    AUCUN_CLIC,
    CLIC_DROIT,
    CLIC_GAUCHE,
    LEFT_CLICK,
    RIGHT_CLICK,
    RemoteController,
    detecter_action,
)


def test_detecter_action_compat():
    """Vérifie la fonction stateless detecter_action()."""
    assert detecter_action(63277) == CLIC_GAUCHE   # macOS Page Down
    assert detecter_action(63276) == CLIC_DROIT    # macOS Page Up
    assert detecter_action(65366) == CLIC_GAUCHE   # Linux Page Down
    assert detecter_action(65365) == CLIC_DROIT    # Linux Page Up
    assert detecter_action(2228224) == CLIC_GAUCHE # Windows Page Down
    assert detecter_action(2162688) == CLIC_DROIT  # Windows Page Up
    assert detecter_action(ord("a")) is AUCUN_CLIC
    assert detecter_action(-1) is AUCUN_CLIC
    print("OK  compatibilite detecter_action multi-OS")


def test_no_trigger_without_press():
    """Vérifie qu'aucun événement n'est généré sans appui ou sur touche inconnue."""
    ctrl = RemoteController(debounce_ms=100)
    assert ctrl.process_opencv_key(-1, now=1.0) is None
    assert ctrl.process_opencv_key(ord("x"), now=1.1) is None
    assert ctrl.process_keysym("", now=1.2) is None
    assert ctrl.process_keysym("space", now=1.3) is None
    assert ctrl.get_stats() == {
        "total_next": 0,
        "total_previous": 0,
        "ignored_debounce": 0,
    }
    print("OK  aucun declenchement sans appui")


def test_distinction_next_previous():
    """Vérifie la distinction claire entre Next (LEFT_CLICK) et Previous (RIGHT_CLICK)."""
    ctrl = RemoteController(debounce_ms=0)

    ev_next = ctrl.process_opencv_key(63277, now=10.0)
    assert ev_next is not None
    assert ev_next.action == LEFT_CLICK
    assert ev_next.button == "NEXT"
    assert ev_next.source == "hid"

    ev_prev = ctrl.process_opencv_key(63276, now=10.1)
    assert ev_prev is not None
    assert ev_prev.action == RIGHT_CLICK
    assert ev_prev.button == "PREVIOUS"
    assert ev_prev.source == "hid"

    ev_sym_next = ctrl.process_keysym("Page_Down", now=10.2)
    ev_sym_prev = ctrl.process_keysym("Page_Up", now=10.3)
    assert ev_sym_next is not None and ev_sym_next.action == LEFT_CLICK
    assert ev_sym_prev is not None and ev_sym_prev.action == RIGHT_CLICK
    print("OK  distinction Next (clic gauche) et Previous (clic droit)")


def test_rapid_presses_and_debounce():
    """Vérifie l'anti-rebond et la prise en compte des appuis successifs légitimes."""
    ctrl = RemoteController(debounce_ms=150)

    # 1er appui à t=1.00 -> accepté
    assert ctrl.process_opencv_key(63277, now=1.00) is not None
    # Répétition à t=1.05 (< 150ms) -> filtrée
    assert ctrl.process_opencv_key(63277, now=1.05) is None
    # Un autre bouton à t=1.06 -> accepté immédiatement (anti-rebond par bouton)
    assert ctrl.process_opencv_key(63276, now=1.06) is not None
    # 2e appui NEXT à t=1.20 (>= 150ms) -> accepté
    assert ctrl.process_opencv_key(63277, now=1.20) is not None

    stats = ctrl.get_stats()
    assert stats["total_next"] == 2
    assert stats["total_previous"] == 1
    assert stats["ignored_debounce"] == 1
    print("OK  appuis successifs et anti-rebond")


def test_keyboard_fallback_toggle():
    """Vérifie que les raccourcis clavier 'n'/'p' peuvent être activés ou désactivés."""
    ctrl_off = RemoteController(enable_keyboard_fallback=False)
    assert ctrl_off.process_opencv_key(ord("n"), now=1.0) is None
    assert ctrl_off.process_opencv_key(ord("p"), now=1.1) is None

    ctrl_on = RemoteController(enable_keyboard_fallback=True)
    ev = ctrl_on.process_opencv_key(ord("n"), now=2.0)
    assert ev is not None and ev.action == LEFT_CLICK and ev.source == "keyboard_fallback"
    print("OK  raccourcis clavier de secours configurables")


def main():
    test_detecter_action_compat()
    test_no_trigger_without_press()
    test_distinction_next_previous()
    test_rapid_presses_and_debounce()
    test_keyboard_fallback_toggle()
    print("\nTous les tests P3 sont passes.")


if __name__ == "__main__":
    main()
