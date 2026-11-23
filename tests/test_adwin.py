from griddrift.drift import ADWINConfig, PersistentADWIN


def test_persistent_adwin_accepts_the_configured_error_stream():
    detector = PersistentADWIN(ADWINConfig(delta=0.002, clock=32, min_window_length=5))
    changes = []
    for error in [10.0] * 128 + [1000.0] * 256:
        if detector.update(error):
            changes.append(detector.last_change)
    assert detector.observations == 384
    assert changes
    change = changes[0]
    assert change is not None
    assert change.error_mean_after_mw > change.error_mean_before_mw
    assert change.direction == "error_increased"
