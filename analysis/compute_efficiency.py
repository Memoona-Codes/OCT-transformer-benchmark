"""
Computes Params and Inference Time for a model at a given config, matching
the paper's Section 2.4 protocol EXACTLY:
  - measured on the constructed (UNTRAINED) model -- params/inference time
    are architecture properties, not training outcomes
  - inference time = mean +/- std across 5 repeated forward passes on a
    batch of 100 samples
  - ONE excluded warm-up call first (to avoid counting first-call graph-
    tracing overhead)
  - uses the model's fold-3 best-found configuration (matching how
    Retformer/ViT/Swin/KAN's Table 3 numbers were produced)

Usage:
    python3 compute_efficiency.py deit
    python3 compute_efficiency.py mae
"""

import sys
import time
import numpy as np
import keras

INPUT_SHAPE = (50, 50, 1)
BATCH_SIZE = 100
N_TIMED_CALLS = 5

# Fold-3 best-found configs (from the completed full runs)
FOLD3_CONFIGS = {
    "deit": dict(proj_dim=58, patch_size=5, layers_n=1, heads=2, mlp_head_1=32, mlp_head_2=32),
    "mae": dict(proj_dim=52, patch_size=10, layers_n=2, heads=5, mlp_head_1=32, mlp_head_2=16),
}


def count_params(model):
    return model.count_params()


def measure_inference_time(model, batch_size=BATCH_SIZE, n_calls=N_TIMED_CALLS):
    dummy_input = np.random.rand(batch_size, *INPUT_SHAPE).astype(np.float32)

    # one excluded warm-up call (graph tracing overhead)
    _ = model.predict(dummy_input, verbose=0)

    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def main(model_key):
    config = FOLD3_CONFIGS.get(model_key)
    if config is None:
        print(f"No fold-3 config recorded yet for '{model_key}'. "
              f"Fill in FOLD3_CONFIGS['{model_key}'] with the best params "
              f"printed at the end of its full run, then re-run this script.")
        sys.exit(1)

    if model_key == "deit":
        from deit_niter15_v3 import build_deit
        model = build_deit(INPUT_SHAPE, **config)
    elif model_key == "mae":
        from mae_niter15_v3 import build_mae_encoder
        model = build_mae_encoder(INPUT_SHAPE, **config)
    else:
        print(f"Unknown model_key: {model_key}")
        sys.exit(1)

    params = count_params(model)
    mean_time, std_time = measure_inference_time(model)

    print(f"\n=== {model_key.upper()} efficiency (fold-3 config: {config}) ===")
    print(f"Params: {params:,}")
    print(f"Inference Time (100 samples), s: {mean_time:.4f} +/- {std_time:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("deit", "mae"):
        print("Usage: python3 compute_efficiency.py [deit|mae]")
        sys.exit(1)
    main(sys.argv[1])
