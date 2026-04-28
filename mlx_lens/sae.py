"""Sparse Autoencoder support — load and run SAEs on MLX.

Currently supports JumpReLU SAEs (GemmaScope format).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


class JumpReLUSAE:
    """JumpReLU Sparse Autoencoder (Rajamanoharan et al. 2024).

    Supports GemmaScope weight format (``w_enc``, ``w_dec``, ``b_enc``,
    ``b_dec``, ``threshold`` in a safetensors file).

    Usage::

        sae = JumpReLUSAE.from_pretrained("path/to/params.safetensors")
        acts = sae.encode(residual)       # (*, d_model) -> (*, d_sae)
        recon = sae.decode(acts)           # (*, d_sae)  -> (*, d_model)
        directions = sae.directions([42, 710, 145])  # unit decoder dirs
    """

    def __init__(
        self,
        w_enc: mx.array,
        w_dec: mx.array,
        b_enc: mx.array,
        b_dec: mx.array,
        threshold: mx.array,
    ):
        self.w_enc = w_enc          # (d_model, d_sae)
        self.w_dec = w_dec          # (d_sae, d_model)
        self.b_enc = b_enc          # (d_sae,)
        self.b_dec = b_dec          # (d_model,)
        self.threshold = threshold  # (d_sae,)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> JumpReLUSAE:
        """Load from a GemmaScope safetensors file."""
        weights = mx.load(str(path))
        return cls(
            w_enc=weights["w_enc"],
            w_dec=weights["w_dec"],
            b_enc=weights["b_enc"],
            b_dec=weights["b_dec"],
            threshold=weights["threshold"],
        )

    @classmethod
    def from_gemma_scope(
        cls,
        model_id: str = "google/gemma-scope-2-27b-it",
        layer: int = 16,
        width: str = "16k",
        variant: str = "l0_small",
    ) -> JumpReLUSAE:
        """Download and load a GemmaScope SAE from HuggingFace.

        Requires ``huggingface-hub`` (install with ``pip install mlx-lens[huggingface]``).
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for from_gemma_scope(). "
                "Install with: pip install mlx-lens[huggingface]"
            )
        filename = f"resid_post/layer_{layer}_width_{width}_{variant}/params.safetensors"
        local_path = hf_hub_download(repo_id=model_id, filename=filename)
        return cls.from_pretrained(local_path)

    # ── Properties ──────────────────────────────────────────

    @property
    def d_model(self) -> int:
        return self.w_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

    # ── Core ops ────────────────────────────────────────────

    def encode(self, x: mx.array) -> mx.array:
        """Encode residual stream into SAE feature activations.

        Args:
            x: ``(*, d_model)`` residual stream tensor.

        Returns:
            ``(*, d_sae)`` sparse activation tensor.
        """
        pre = x @ self.w_enc + self.b_enc
        return mx.maximum(pre, 0) * (pre > self.threshold)

    def decode(self, z: mx.array) -> mx.array:
        """Reconstruct residual from SAE activations.

        Args:
            z: ``(*, d_sae)`` activation tensor.

        Returns:
            ``(*, d_model)`` reconstructed residual.
        """
        return z @ self.w_dec + self.b_dec

    def directions(self, feature_indices: list[int] | None = None) -> mx.array:
        """Unit-normalized decoder directions.

        Args:
            feature_indices: specific features to return. If ``None``,
                returns all ``d_sae`` directions.

        Returns:
            ``(n_features, d_model)`` unit vectors.
        """
        if feature_indices is not None:
            vecs = self.w_dec[mx.array(feature_indices)]
        else:
            vecs = self.w_dec
        norms = mx.linalg.norm(vecs, axis=-1, keepdims=True)
        return vecs / (norms + 1e-8)
