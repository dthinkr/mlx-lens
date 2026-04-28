"""Scale search: find the steering scale that produces a target loss increase."""

import mlx.core as mx
from mlx_lens import LensModel

lens = LensModel("mlx-community/gemma-3-1b-it-4bit")

# A unit-norm steering direction (replace with an SAE feature direction)
d = lens.d_model
direction = mx.random.normal((d,))
direction = direction / mx.linalg.norm(direction)

# Text samples for measuring loss
texts = [
    "The stock market opened higher on Monday",
    "Interest rates remain unchanged after the meeting",
    "New research suggests that climate change",
]
token_lists = [lens.tokenizer.encode(t) for t in texts]


def avg_loss(scale: float) -> float:
    """Average cross-entropy loss across samples at a given steering scale."""
    losses = []
    for tokens in token_lists:
        if scale == 0:
            losses.append(lens.loss(tokens))
        else:
            with lens.steer(layer=8, vector=direction, scale=scale):
                losses.append(lens.loss(tokens))
    return sum(losses) / len(losses)


# --- Grid search ---------------------------------------------------------
print("Scale-loss curve:")
for scale in [0, 50, 100, 200, 500, 1000]:
    loss = avg_loss(scale)
    print(f"  scale={scale:5d}  loss={loss:.4f}")


# --- Binary search for target loss --------------------------------------
def find_scale(target_loss: float, lo: float = 0, hi: float = 5000) -> float:
    for _ in range(15):
        mid = (lo + hi) / 2
        if avg_loss(mid) < target_loss:
            lo = mid
        else:
            hi = mid
        if hi - lo < 10:
            break
    return (lo + hi) / 2


target = 5.0
optimal_scale = find_scale(target)
print(f"\nOptimal scale for target_loss={target}: {optimal_scale:.0f}")
print(f"  Verification: loss={avg_loss(optimal_scale):.4f}")
