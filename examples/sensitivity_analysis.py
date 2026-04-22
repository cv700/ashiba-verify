"""Sensitivity analysis for Bug 2A — characterize the Freivalds detection floor.

For a fixed matrix shape and fixed k, sweep corruption magnitude and measure the
empirical detection rate. Produces the sensitivity curve that Bug 2A's trade-off
depends on, with enough resolution to identify the crossover from "never catches"
→ "probabilistic" → "always catches".

Output: JSON-Lines to stdout.
"""

import argparse
import json

import torch

from ashiba_verify import freivalds_verify


def sweep_single_element_corruption(
    device,
    dtype,
    m=256, n=128, p=64,
    k=20,
    n_trials_per_magnitude=50,
    seed=None,
):
    """Sweep single-element corruption magnitude. One element of C perturbed by delta.

    Detection rate should rise sigmoidally from 0 (below floor) to 1 (well above floor).
    """
    # Build fixed A, B so the corrupt C is deterministic per trial.
    # But regenerate r each trial to measure ensemble detection probability.
    if seed is not None:
        torch.manual_seed(seed)

    # Calibrate: what is the norm-based threshold at this shape?
    A = torch.randn(m, n, device=device, dtype=dtype)
    B = torch.randn(n, p, device=device, dtype=dtype)
    C = A @ B

    # Use default tolerance (atol=rtol=1e-4 for fp32)
    # Estimate cr_max empirically
    r_probe = torch.empty(p, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    cr_probe = (C @ r_probe).abs().max().item()
    threshold_estimate = 1e-4 + 1e-4 * cr_probe

    print(json.dumps({
        "event": "calibration",
        "m": m, "n": n, "p": p,
        "k": k,
        "cr_max_estimate": round(cr_probe, 3),
        "threshold_estimate": round(threshold_estimate, 6),
        "theoretical_floor_single_element": round(2 * threshold_estimate, 6),
    }))

    # Magnitudes to sweep (log-spaced from 10x below threshold to 100x above)
    magnitudes = [
        threshold_estimate * factor
        for factor in [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
    ]

    for mag in magnitudes:
        detections = 0
        for trial in range(n_trials_per_magnitude):
            C_corrupt = C.clone()
            # Pick a random position each trial (so we sweep positions too)
            i_idx = torch.randint(0, m, (1,)).item()
            j_idx = torch.randint(0, p, (1,)).item()
            C_corrupt[i_idx, j_idx] += mag

            # freivalds_verify returns True if it passes (verifier says correct).
            # A corrupted matmul should return False (verifier detects).
            result = freivalds_verify(A, B, C_corrupt, k=k)
            if result is False:
                detections += 1

        detection_rate = detections / n_trials_per_magnitude
        print(json.dumps({
            "event": "measurement",
            "corruption_type": "single_element",
            "magnitude": round(mag, 6),
            "magnitude_vs_threshold": round(mag / threshold_estimate, 3),
            "detection_rate": round(detection_rate, 3),
            "n_trials": n_trials_per_magnitude,
        }))


def sweep_uniform_noise_corruption(
    device,
    dtype,
    m=256, n=128, p=64,
    k=20,
    n_trials_per_magnitude=50,
    seed=None,
):
    """Sweep uniform-noise corruption. Every element of C perturbed by N(0, sigma).

    Much easier to detect than single-element — noise accumulates across elements.
    """
    if seed is not None:
        torch.manual_seed(seed)

    A = torch.randn(m, n, device=device, dtype=dtype)
    B = torch.randn(n, p, device=device, dtype=dtype)
    C = A @ B

    r_probe = torch.empty(p, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    cr_probe = (C @ r_probe).abs().max().item()
    threshold_estimate = 1e-4 + 1e-4 * cr_probe

    sigmas = [
        threshold_estimate * factor
        for factor in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    ]

    for sigma in sigmas:
        detections = 0
        for trial in range(n_trials_per_magnitude):
            C_corrupt = C + sigma * torch.randn_like(C)
            result = freivalds_verify(A, B, C_corrupt, k=k)
            if result is False:
                detections += 1
        detection_rate = detections / n_trials_per_magnitude
        print(json.dumps({
            "event": "measurement",
            "corruption_type": "uniform_noise",
            "sigma": round(sigma, 6),
            "sigma_vs_threshold": round(sigma / threshold_estimate, 3),
            "detection_rate": round(detection_rate, 3),
            "n_trials": n_trials_per_magnitude,
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    print(json.dumps({
        "event": "run_start",
        "device": str(device),
        "k": args.k,
        "n_trials_per_magnitude": args.n_trials,
        "seed": args.seed,
    }))

    print(json.dumps({"event": "sweep_start", "corruption_type": "single_element"}))
    sweep_single_element_corruption(
        device, dtype,
        k=args.k,
        n_trials_per_magnitude=args.n_trials,
        seed=args.seed,
    )

    print(json.dumps({"event": "sweep_start", "corruption_type": "uniform_noise"}))
    sweep_uniform_noise_corruption(
        device, dtype,
        k=args.k,
        n_trials_per_magnitude=args.n_trials,
        seed=args.seed,
    )

    print(json.dumps({"event": "run_end"}))


if __name__ == "__main__":
    main()
