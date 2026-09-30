"""D-EMA decay calculator: drift budget <-> decay and the scale-normalized horizon."""

import math

import pytest

from dynthinker.tools.ema_decay import adapt_beta, beta_from_drift, compute_ema_updates, drift_from_beta


def test_update_count_matches_training_loop():
    steps, updates = compute_ema_updates(2_590_665, batch_size=64, start_step=150, update_interval=10)
    assert steps == 40480 and updates == 4034


def test_budget_round_trip_and_example_decay():
    _, updates = compute_ema_updates(2_590_665)
    beta = beta_from_drift(0.125, updates)
    assert round(beta, 6) == 0.999967
    assert drift_from_beta(beta, updates) == pytest.approx(0.125)


def test_scale_normalized_horizon_preserves_budget():
    new_beta, old_drift, _ = adapt_beta(0.999551, 200_000, 1_000_000)
    _, n_old = compute_ema_updates(200_000)
    _, n_new = compute_ema_updates(1_000_000)
    assert old_drift == pytest.approx(1 - 0.999551**n_old)
    assert new_beta == pytest.approx(math.exp(n_old * math.log(0.999551) / n_new))
    assert drift_from_beta(new_beta, n_new) == pytest.approx(old_drift)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        beta_from_drift(1.0, 10)
    with pytest.raises(ValueError):
        beta_from_drift(0.1, 0)


def test_callback_reports_realised_drift():
    from dynthinker.train.dema import DecoupledEMACallback

    cb = DecoupledEMACallback(mlp_decay=0.999552, other_decay=0.999552, start_step=150, update_interval=10)
    updates, drift = cb.planned_drift(3125)  # 200k samples / 64
    assert updates == compute_ema_updates(200_000)[1] == 298
    assert drift["mlp"] == pytest.approx(drift_from_beta(0.999552, 298))
    assert cb.planned_drift(3120)[0] == 298 and cb.planned_drift(3119)[0] == 297
