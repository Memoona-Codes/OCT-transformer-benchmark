"""
Efficiency comparison -- parameter count and inference time for all four
models, at each model's own fold-3 best-found configuration. Does NOT
retrain anything: builds each architecture directly (bypassing the
sklearn wrapper's .fit(), which would trigger a full 100-epoch training
run) and measures the untrained model directly, since parameter count and
inference latency are architecture properties independent of training.

Run this ON MOORE from ~/retformer_paper_repro:
    conda activate retformer_repro
    python -u measure_efficiency.py 2>&1 | tee efficiency_output.txt

Requires: retformer.py and
vit_swin_kfold_nested_matched.py all present in the current directory
(their model-building functions are imported directly, not reimplemented,
to avoid any risk of mismatch).
"""
import time
import numpy as np

INPUT_SHAPE = (50, 50, 1)
N_INFER_SAMPLES = 100   # batch size for timing the inference pass
N_INFER_REPEATS = 5     # repeat the timed predict() call this many times, report mean

# Dummy input for inference timing -- architecture/timing is independent of
# actual pixel content, so random noise is sufficient here.
dummy_input = np.random.rand(N_INFER_SAMPLES, *INPUT_SHAPE).astype("float32")


def time_inference(model, x, flat=False):
    """Times N_INFER_REPEATS predict() calls, returns (mean, std) in seconds."""
    x_in = x.reshape(x.shape[0], -1) if flat else x
    # one warm-up call (excluded from timing) to avoid counting first-call
    # graph-tracing overhead as inference cost
    _ = model.predict(x_in, verbose=0)
    times = []
    for _ in range(N_INFER_REPEATS):
        t0 = time.time()
        _ = model.predict(x_in, verbose=0)
        times.append(time.time() - t0)
    return float(np.mean(times)), float(np.std(times))


results = []

# ---------------------------------------------------------------------
# Retformer -- fold 3 best config
# ---------------------------------------------------------------------
print("=== Retformer ===")
from retformer import create_vit_classifier
retformer_model = create_vit_classifier(
    INPUT_SHAPE,
    projection_dim=56, patch_size=4, transformer_layers=1,
    num_q_heads=3, num_kv_heads=2,
    mlp_head_unit_1=40, mlp_head_unit_2=19,
)
retformer_params = retformer_model.count_params()
retformer_mean, retformer_std = time_inference(retformer_model, dummy_input)
print(f"Params: {retformer_params:,}")
print(f"Inference time ({N_INFER_SAMPLES} samples): {retformer_mean:.4f}s +/- {retformer_std:.4f}s")
results.append(("Retformer", retformer_params, retformer_mean, retformer_std))

# ---------------------------------------------------------------------
# ViT -- fold 3 best config
# ---------------------------------------------------------------------
print("\n=== ViT ===")
from vit_swin_kfold_nested_matched_v2 import create_vanilla_vit
vit_model = create_vanilla_vit(
    INPUT_SHAPE,
    projection_dim=36, patch_size=5, transformer_layers=2,
    num_heads=5, mlp_head_unit_1=54, mlp_head_unit_2=32,
)
vit_params = vit_model.count_params()
vit_mean, vit_std = time_inference(vit_model, dummy_input)
print(f"Params: {vit_params:,}")
print(f"Inference time ({N_INFER_SAMPLES} samples): {vit_mean:.4f}s +/- {vit_std:.4f}s")
results.append(("ViT", vit_params, vit_mean, vit_std))

# ---------------------------------------------------------------------
# Swin -- fold 3 best config
# ---------------------------------------------------------------------
print("\n=== Swin ===")
from vit_swin_kfold_nested_matched_v2 import create_swin_flat
swin_model = create_swin_flat(
    INPUT_SHAPE,
    projection_dim=60, patch_size=5, transformer_layers=3,
    num_heads=5, window_size=5,
    mlp_head_unit_1=34, mlp_head_unit_2=18,
)
swin_params = swin_model.count_params()
swin_mean, swin_std = time_inference(swin_model, dummy_input)
print(f"Params: {swin_params:,}")
print(f"Inference time ({N_INFER_SAMPLES} samples): {swin_mean:.4f}s +/- {swin_std:.4f}s")
results.append(("Swin", swin_params, swin_mean, swin_std))

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------
print("\n\n=== SUMMARY ===")
print(f"{'Model':<12}{'Params':>14}{'Inference (s)':>20}")
for name, params, mean, std in results:
    print(f"{name:<12}{params:>14,}{mean:>14.4f} +/- {std:.4f}")
