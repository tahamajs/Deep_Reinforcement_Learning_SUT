from planner.triggers import should_trigger, TriggerConfig


def test_cooldown_and_thresholds():
    cfg = TriggerConfig(
        cooldown=5,
        trigger_td=0.5,
        trigger_unc=0.2,
        trigger_ent_low=0.1,
        trigger_ent_high=1.0,
    )
    # no last trigger -> should fire on td_error
    assert should_trigger(0.6, 0.0, 0.5, cfg, last_trigger=None, step=10)
    last = 10
    # within cooldown -> should not fire
    assert not should_trigger(0.6, 0.0, 0.5, cfg, last_trigger=last, step=12)
    # after cooldown -> should fire
    assert should_trigger(0.6, 0.0, 0.5, cfg, last_trigger=last, step=16)
















