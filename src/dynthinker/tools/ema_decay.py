# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""D-EMA decay calculator (`dynthinker-ema-decay`).

For a group with decay ``rho`` and ``N`` EMA updates, the shadow after training is
``rho^N * theta_0 + (1 - rho) * sum_t rho^(N-t) * theta_t``. The *target drift budget*
``delta = 1 - rho^N`` is the total weight on post-training states, so
``rho = (1 - delta)^(1 / N)``. Keeping ``delta`` fixed across schedules gives the scale-normalized
horizon ``rho' = rho^(N / N')``.

The number of updates is estimated from ``steps = ceil(dataset_size * epochs / batch)`` optimizer steps,
with updates at ``start_step, start_step + interval, ...`` (inclusive), counted by
``dynthinker.train.dema.expected_update_count``. This is an approximation: the HF Trainer runs
``floor(len(dataloader) / grad_accum)`` steps per epoch, and the modality-grouped sampler drops each tag's
last incomplete mega-batch, so the realised step count (and drift) can be slightly lower. The D-EMA
callback logs the realised update count and per-group drift from the Trainer's ``max_steps`` at train begin.

Examples:
    dynthinker-ema-decay --dataset_size 200000 --target_drift 0.125
    dynthinker-ema-decay --dataset_size 498000 --beta 0.9998
    dynthinker-ema-decay --dataset_size 1000000 --old_dataset_size 200000 --old_beta 0.999551
"""

import argparse
import math
import sys
from typing import Tuple


def compute_ema_updates(
    dataset_size: int,
    batch_size: int = 64,
    start_step: int = 150,
    update_interval: int = 10,
    num_epochs: int = 1,
) -> Tuple[int, int]:
    """Total optimizer steps and number of D-EMA updates."""
    from ..train.dema import expected_update_count

    total_steps = math.ceil(dataset_size * num_epochs / batch_size)
    return total_steps, expected_update_count(total_steps, start_step, update_interval)


def beta_from_drift(target_drift: float, ema_updates: int) -> float:
    """rho = (1 - delta)^(1/N)."""
    if ema_updates <= 0:
        raise ValueError("EMA updates <= 0, check dataset_size / start_step / interval")
    if not 0 < target_drift < 1:
        raise ValueError(f"target_drift must be in (0, 1), got {target_drift}")
    return (1.0 - target_drift) ** (1.0 / ema_updates)


def drift_from_beta(beta: float, ema_updates: int) -> float:
    """delta = 1 - rho^N."""
    if ema_updates <= 0:
        return 0.0
    return 1.0 - beta**ema_updates


def adapt_beta(old_beta, old_dataset_size, new_dataset_size, batch_size=64, start_step=150, update_interval=10):
    """Keep the drift budget of (old_beta, old schedule) on a new dataset size."""
    _, t_old = compute_ema_updates(old_dataset_size, batch_size, start_step, update_interval)
    _, t_new = compute_ema_updates(new_dataset_size, batch_size, start_step, update_interval)
    old_drift = drift_from_beta(old_beta, t_old)
    return beta_from_drift(old_drift, t_new), old_drift, drift_from_beta(old_beta, t_new)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dynthinker-ema-decay", description="D-EMA decay / drift-budget calculator")
    parser.add_argument("--dataset_size", type=int, required=True, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Global batch size")
    parser.add_argument("--start_step", type=int, default=150)
    parser.add_argument("--update_interval", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--target_drift", type=float, nargs="+", default=None, help="Drift budget(s), e.g. 0.125")
    parser.add_argument("--beta", type=float, nargs="+", default=None, help="Decay value(s) to evaluate")
    parser.add_argument("--old_beta", type=float, default=None, help="Decay to transfer to a new dataset size")
    parser.add_argument("--old_dataset_size", type=int, default=None)
    args = parser.parse_args(argv)

    total_steps, ema_updates = compute_ema_updates(
        args.dataset_size, args.batch_size, args.start_step, args.update_interval, args.num_epochs
    )
    print(f"dataset_size={args.dataset_size} batch_size={args.batch_size} epochs={args.num_epochs}")
    print(f"total_steps={total_steps} start_step={args.start_step} interval={args.update_interval} ema_updates={ema_updates}")

    if args.old_beta is not None and args.old_dataset_size is not None:
        new_beta, old_drift, unchanged = adapt_beta(
            args.old_beta, args.old_dataset_size, args.dataset_size, args.batch_size, args.start_step, args.update_interval
        )
        print(f"old beta {args.old_beta}: drift {old_drift * 100:.2f}% on the old schedule")
        print(f"same beta on the new schedule: drift {unchanged * 100:.2f}%")
        print(f"recommended beta: {new_beta:.6f} (drift {drift_from_beta(new_beta, ema_updates) * 100:.2f}%)")
        return 0
    if args.target_drift is not None:
        for drift in args.target_drift:
            beta = beta_from_drift(drift, ema_updates)
            print(f"drift {drift * 100:.2f}% -> beta {beta:.6f}")
        return 0
    betas = args.beta if args.beta is not None else [0.9999, 0.9998, 0.9997, 0.9996, 0.9995, 0.999, 0.998]
    for beta in betas:
        drift = drift_from_beta(beta, ema_updates)
        print(f"beta {beta:.6f} -> drift {drift * 100:.2f}% (initialization weight {(1 - drift) * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
