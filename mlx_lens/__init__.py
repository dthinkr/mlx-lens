"""mlx-lens: Mechanistic interpretability on Apple Silicon.

Steering vectors, residual capture, and SAE analysis for MLX models.
The Mac-native equivalent of vLLM-Lens.
"""

from mlx_lens.model import CaptureResult, LensModel
from mlx_lens.sae import JumpReLUSAE

__version__ = "0.1.0"
__all__ = ["LensModel", "JumpReLUSAE", "CaptureResult"]
