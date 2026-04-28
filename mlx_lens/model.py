"""LensModel: load an MLX model and intervene on its residual stream.

Core capabilities:
  - steer(layer, vector, scale): inject steering vectors at any decoder layer
  - capture(layers): read residual stream activations
  - loss(tokens): compute next-token cross-entropy (for scale search)
  - generate(prompt, ...): text generation with active steering
  - engine-backed batch generation via vllm-mlx (optional)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate as _mlx_generate
from mlx_lm import load as _mlx_load


# ── Layer wrappers ──────────────────────────────────────────


class _SteeredLayer:
    """Proxy that adds a scaled steering vector to a layer's output."""

    __slots__ = ("_original", "_vector", "_scale")

    def __init__(self, original, vector: mx.array, scale: float):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_vector", vector)
        object.__setattr__(self, "_scale", scale)

    def __call__(self, x, *args, **kwargs):
        out = self._original(x, *args, **kwargs)
        return out + self._scale * self._vector

    def __getattr__(self, name):
        return getattr(self._original, name)


class _CaptureLayer:
    """Proxy that records a layer's output while passing it through."""

    __slots__ = ("_original", "output")

    def __init__(self, original):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "output", None)

    def __call__(self, x, *args, **kwargs):
        out = self._original(x, *args, **kwargs)
        object.__setattr__(self, "output", out)
        return out

    def __getattr__(self, name):
        return getattr(self._original, name)


class _SteeredCaptureLayer:
    """Proxy that steers AND captures (for nested steer + capture)."""

    __slots__ = ("_original", "_vector", "_scale", "output")

    def __init__(self, original, vector: mx.array, scale: float):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_vector", vector)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "output", None)

    def __call__(self, x, *args, **kwargs):
        out = self._original(x, *args, **kwargs)
        out = out + self._scale * self._vector
        object.__setattr__(self, "output", out)
        return out

    def __getattr__(self, name):
        return getattr(self._original, name)


# ── Capture result container ────────────────────────────────


class CaptureResult:
    """Dict-like container for captured residual streams."""

    def __init__(self):
        self._data: dict[int, mx.array] = {}

    def __getitem__(self, layer: int) -> mx.array:
        return self._data[layer]

    def __contains__(self, layer: int) -> bool:
        return layer in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __repr__(self):
        shapes = {k: v.shape for k, v in self._data.items()}
        return f"CaptureResult({shapes})"


# ── Main model class ────────────────────────────────────────


class LensModel:
    """MLX model with residual-stream intervention capabilities.

    Usage::

        lens = LensModel("mlx-community/gemma-3-27b-it-qat-4bit")

        # Steer and compute loss
        with lens.steer(layer=16, vector=vec, scale=1000):
            l = lens.loss(tokens)

        # Steer and generate
        with lens.steer(layer=16, vector=vec, scale=1000):
            text = lens.generate("What is money?", max_tokens=200)

        # Capture residual stream
        with lens.capture(layers=[16]) as cap:
            lens.forward(tokens)
        residual = cap[16]  # (seq_len, d_model)
    """

    def __init__(self, model_path: str, use_engine: bool = False, **load_kwargs):
        self._model, self._tokenizer = _mlx_load(model_path, **load_kwargs)
        self._layers = _find_layers(self._model)
        self._d_model = _detect_d_model(self._model)
        self._active_steers: dict[int, tuple] = {}
        self._engine = None
        if use_engine:
            self._init_engine()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def d_model(self) -> int | None:
        return self._d_model

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def engine(self):
        if self._engine is None:
            self._init_engine()
        return self._engine

    def _init_engine(self):
        try:
            from vllm_mlx import EngineCore, EngineConfig
        except ImportError:
            raise ImportError(
                "vllm-mlx is required for engine-backed generation. "
                "Install with: pip install 'mlx-lens[engine]'"
            )
        self._engine = EngineCore(self._model, self._tokenizer, EngineConfig())

    # ── Intervention API ────────────────────────────────────

    @contextmanager
    def steer(self, layer: int, vector: mx.array, scale: float):
        """Temporarily add a scaled steering vector at ``layer``.

        Can be nested with :meth:`capture` — the captured output will
        include the steering effect.
        """
        if not isinstance(vector, mx.array):
            vector = mx.array(vector)
        original = self._layers[layer]
        self._active_steers[layer] = (vector, scale)
        self._layers[layer] = _SteeredLayer(original, vector, scale)
        try:
            yield
        finally:
            self._layers[layer] = original
            self._active_steers.pop(layer, None)

    @contextmanager
    def capture(self, layers: Sequence[int]):
        """Temporarily capture residual-stream activations at ``layers``.

        Yields a :class:`CaptureResult` populated after :meth:`forward`
        or :meth:`generate` completes inside the block.
        """
        result = CaptureResult()
        originals = {}
        wrappers = {}

        for layer_idx in layers:
            current = self._layers[layer_idx]
            originals[layer_idx] = current

            if layer_idx in self._active_steers:
                vec, sc = self._active_steers[layer_idx]
                real_original = current._original if isinstance(current, _SteeredLayer) else current
                wrapper = _SteeredCaptureLayer(real_original, vec, sc)
            else:
                wrapper = _CaptureLayer(current)

            wrappers[layer_idx] = wrapper
            self._layers[layer_idx] = wrapper

        try:
            yield result
        finally:
            for layer_idx in layers:
                w = wrappers[layer_idx]
                if w.output is not None:
                    result._data[layer_idx] = w.output
                self._layers[layer_idx] = originals[layer_idx]

    # ── Forward / loss / generate ───────────────────────────

    def forward(self, tokens) -> mx.array:
        """Raw forward pass. Returns logits ``(batch, seq_len, vocab)``."""
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        return self._model(tokens)

    def loss(self, tokens) -> float:
        """Mean next-token cross-entropy loss."""
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        logits = self._model(tokens)
        targets = tokens[:, 1:]
        logits = logits[:, :-1, :]
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_lp = mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        mx.eval(target_lp)
        return -mx.mean(target_lp).item()

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text, respecting any active steering."""
        return _mlx_generate(self._model, self._tokenizer, prompt=prompt, **kwargs)

    def generate_batch(self, prompts: list[str], max_tokens: int = 256,
                       temperature: float = 0.0, **kwargs) -> list[str]:
        """Batch generation via vllm-mlx engine. Respects active steering."""
        from vllm_mlx import SamplingParams as VllmSP
        sp = VllmSP(max_tokens=max_tokens, temperature=temperature, **kwargs)
        results = self.engine.generate_batch_sync(prompts, sp)
        return [r.output_text for r in results]

    def close(self):
        """Release engine resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None


# ── Helpers ─────────────────────────────────────────────────


_LAYER_PATHS = [
    lambda m: m.language_model.model.layers,  # multimodal (gemma3)
    lambda m: m.model.layers,                  # text-only
    lambda m: m.model.decoder.layers,          # encoder-decoder
    lambda m: m.layers,                        # flat
]


def _find_layers(model) -> list:
    """Duck-typed decoder layer discovery (mirrors vLLM-Lens)."""
    for accessor in _LAYER_PATHS:
        try:
            layers = accessor(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(
        "Cannot find decoder layers. Supported layouts: "
        "model.language_model.model.layers, model.model.layers, "
        "model.model.decoder.layers, model.layers"
    )


def _detect_d_model(model) -> int | None:
    """Best-effort detection of hidden size from model config."""
    for attr_chain in [
        ["args", "hidden_size"],
        ["config", "hidden_size"],
        ["language_model", "args", "hidden_size"],
        ["language_model", "config", "hidden_size"],
    ]:
        obj = model
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if isinstance(obj, int):
                return obj
        except AttributeError:
            continue
    return None
