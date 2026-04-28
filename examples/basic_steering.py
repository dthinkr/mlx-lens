"""Basic steering: inject a direction into a decoder layer and observe the effect."""

import mlx.core as mx
from mlx_lens import LensModel

# Load any mlx-lm compatible model
lens = LensModel("mlx-community/gemma-3-1b-it-4bit")

# --- Steering with a random direction -----------------------------------
d = lens.d_model
direction = mx.random.normal((d,))
direction = direction / mx.linalg.norm(direction)

# Baseline
loss_base = lens.loss("The capital of France is Paris")
print(f"Baseline loss: {loss_base:.4f}")

# Steered — scale controls perturbation strength
with lens.steer(layer=8, vector=direction, scale=200.0):
    loss_steered = lens.loss("The capital of France is Paris")
print(f"Steered loss:  {loss_steered:.4f}  (Δ={loss_steered - loss_base:+.4f})")

# Steering is automatically removed outside the context manager
loss_after = lens.loss("The capital of France is Paris")
assert abs(loss_after - loss_base) < 1e-4

# --- Steered generation --------------------------------------------------
print("\nBaseline generation:")
print(lens.generate("What is money?", max_tokens=50))

with lens.steer(layer=8, vector=direction, scale=300.0):
    print("\nSteered generation:")
    print(lens.generate("What is money?", max_tokens=50))
