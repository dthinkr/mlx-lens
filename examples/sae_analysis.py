"""SAE analysis: capture residual streams and decompose them into sparse features."""

import mlx.core as mx
from mlx_lens import JumpReLUSAE, LensModel

lens = LensModel("mlx-community/gemma-3-1b-it-4bit")

# Load a GemmaScope SAE (downloads from HuggingFace on first run)
sae = JumpReLUSAE.from_gemma_scope(
    model_id="google/gemma-scope-2-2b-it",
    layer=12,
    width="16k",
)
print(f"SAE: d_model={sae.d_model}, d_sae={sae.d_sae}")

# --- Capture residual stream at the SAE's layer -------------------------
tokens = lens.tokenizer.encode("Central banks control the money supply")

with lens.capture(layers=[12]) as cap:
    lens.forward(tokens)

residual = cap[12]  # (1, seq_len, d_model)
print(f"Residual shape: {residual.shape}")

# --- Encode into SAE features -------------------------------------------
acts = sae.encode(residual[0])  # (seq_len, d_sae)
n_active = (acts > 0).sum(axis=-1)
print(f"Active features per position: {n_active.tolist()}")

# --- Get decoder directions for specific features -----------------------
feature_ids = [42, 710, 1024]
directions = sae.directions(feature_ids)  # (3, d_model), unit normalized
print(f"Decoder directions shape: {directions.shape}")

# --- Steer with a SAE feature direction ----------------------------------
with lens.steer(layer=12, vector=directions[0], scale=100.0):
    print("\nSteered with feature 42:")
    print(lens.generate("Money is", max_tokens=30))
