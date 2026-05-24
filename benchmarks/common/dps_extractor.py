"""Extractor for Depth Preservation Score (DPS) using streaming covariance."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DPSEvaluator:
    """Evaluates DPS using streaming covariance to avoid OOM for large N.

    This class attaches forward hooks to a given model to capture the output of
    a specific intermediate layer (target) and the final post-LayerNorm state (source).
    It accumulates the cross-products necessary for Ridge Regression incrementally.
    """

    def __init__(self, model: nn.Module, layer_idx: int, final_norm_name: str = "norm") -> None:
        """Initialize the evaluator for a specific layer.

        Args:
            model: The Transformer model.
            layer_idx: The 0-based index of the layer to probe (the target).
            final_norm_name: The name of the final LayerNorm module before the head.
        """
        self.model = model
        self.layer_idx = layer_idx

        # Determine hidden dimension by inspecting the model's config if possible,
        # otherwise we'll lazily initialize accumulators on the first batch.
        self.d: int | None = None
        self.n_tokens: int = 0

        # Accumulators
        self.xtx: torch.Tensor | None = None
        self.xty: torch.Tensor | None = None
        self.yty: torch.Tensor | None = None

        # Accumulators for GPS (Gradient Preservation Score)
        self.xty_gps: torch.Tensor | None = None
        self.yty_gps: torch.Tensor | None = None

        # Accumulators for variance calculation (sum(y) and sum(y^2))
        self.sum_y: torch.Tensor | None = None
        self.sum_y_gps: torch.Tensor | None = None

        # Accumulator for cosine dissimilarity
        self.sum_cos_dissim: float = 0.0

        # Temporary storage for the hook
        self._current_target: torch.Tensor | None = None
        self._current_source: torch.Tensor | None = None

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(final_norm_name)

    def _register_hooks(self, final_norm_name: str) -> None:
        """Register forward hooks to capture activations."""
        # Find the target layer (e.g., model.layers[layer_idx])
        # We need to adapt this slightly depending on the exact model structure in this repo.
        # Assuming typical structure: model.decoder.layers or model.layers

        target_module = None
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            target_module = self.model.decoder.layers[self.layer_idx]
        elif hasattr(self.model, "layers"):
            target_module = self.model.layers[self.layer_idx]
        else:
            raise ValueError("Could not find layers list in model.")

        final_norm_module = None
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, final_norm_name):
            final_norm_module = getattr(self.model.decoder, final_norm_name)
        elif hasattr(self.model, final_norm_name):
            final_norm_module = getattr(self.model, final_norm_name)
        else:
            # Fallback to the last layer's output if no final norm exists
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
                final_norm_module = self.model.decoder.layers[-1]
            elif hasattr(self.model, "layers"):
                final_norm_module = self.model.layers[-1]
            else:
                raise ValueError(f"Could not find final norm '{final_norm_name}' or last layer.")

        def target_hook(module: nn.Module, inputs: tuple, output: torch.Tensor | tuple) -> None:
            # output might be a tuple if the layer returns (hidden_states, past_key_values, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            self._current_target = hidden.detach().float()

        def source_hook(module: nn.Module, inputs: tuple, output: torch.Tensor | tuple) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._current_source = hidden.detach().float()

        self._hooks.append(target_module.register_forward_hook(target_hook))
        self._hooks.append(final_norm_module.register_forward_hook(source_hook))

    def remove_hooks(self) -> None:
        """Remove the registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _init_accumulators(self, d: int, device: torch.device) -> None:
        """Initialize the accumulation matrices lazily."""
        self.d = d
        # X will be augmented with a column of 1s, so dimension is d+1
        self.xtx = torch.zeros((d + 1, d + 1), device=device, dtype=torch.float32)
        self.xty = torch.zeros((d + 1, d), device=device, dtype=torch.float32)
        self.yty = torch.zeros((), device=device, dtype=torch.float32)
        self.sum_y = torch.zeros((d,), device=device, dtype=torch.float32)

        self.xty_gps = torch.zeros((d + 1, d), device=device, dtype=torch.float32)
        self.yty_gps = torch.zeros((), device=device, dtype=torch.float32)
        self.sum_y_gps = torch.zeros((d,), device=device, dtype=torch.float32)

    def process_batch(self, targets: torch.Tensor | None = None) -> None:
        """Process the captured activations for the current batch and update accumulators.

        This must be called immediately after a forward pass before the next one starts.

        Args:
            targets: Optional ground-truth next-token targets for GPS evaluation.
                Shape: (batch, seq) or flattened to matching token count.
        """
        if self._current_target is None or self._current_source is None:
            raise RuntimeError(
                "process_batch called but activations were not captured. "
                "Did you run a forward pass?"
            )

        # Reshape from (batch, seq, d) to (batch * seq, d)
        target = self._current_target.reshape(-1, self._current_target.size(-1))
        source = self._current_source.reshape(-1, self._current_source.size(-1))

        batch_n = target.size(0)
        d = target.size(1)
        device = target.device

        if self.xtx is None:
            self._init_accumulators(d, device)

        # 1. Apply LayerNorm to target to get a_k
        # We manually compute LN since we just want the normalized vector
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, unbiased=False, keepdim=True)
        eps = 1e-5
        y = (target - mean) / torch.sqrt(var + eps)  # This is a_k

        # 2. Augment source with ones to get X_tilde
        ones = torch.ones((batch_n, 1), device=device, dtype=source.dtype)
        x_tilde = torch.cat([source, ones], dim=1)

        # 3. Update streaming covariance matrices for DPS
        self.xtx += x_tilde.T @ x_tilde
        self.xty += x_tilde.T @ y

        # Tr(Y^T Y) is simply the sum of all squared elements in Y
        self.yty += torch.sum(y**2)

        # 4. Update variance accumulators for Y
        self.sum_y += torch.sum(y, dim=0)

        # 5. Update GPS accumulators if targets are provided
        if targets is not None:
            # Locate the language modeling head
            head = None
            if hasattr(self.model, "lm_head"):
                head = self.model.lm_head
            elif hasattr(self.model, "model") and hasattr(self.model.model, "lm_head"):
                head = self.model.model.lm_head
            else:
                raise AttributeError("Could not find lm_head in the model to compute early gradients.")

            with torch.no_grad():
                # Compute early logits using LayerNorm-normalized activation y (which is a_k)
                early_logits = head(y)
                probs = torch.softmax(early_logits, dim=-1)

                # Flatten targets and compute probability difference: e = probs - one_hot(targets)
                targets_flat = targets.reshape(-1)
                e = probs.clone()
                e[torch.arange(batch_n, device=device), targets_flat] -= 1.0

                # Project error vector back using the head weights to get early gradient g
                g = e @ head.weight  # shape: (batch_n, d)

                # Update streaming covariance matrices for GPS
                self.xty_gps += x_tilde.T @ g
                self.yty_gps += torch.sum(g**2)
                self.sum_y_gps += torch.sum(g, dim=0)

        # 6. Update cosine dissimilarity
        # dissim = 1 - cosine_similarity
        cos_sim = torch.nn.functional.cosine_similarity(target, source, dim=-1)
        self.sum_cos_dissim += float(torch.sum(1.0 - cos_sim).item())

        self.n_tokens += batch_n

        # Clear temporary storage to free memory
        self._current_target = None
        self._current_source = None

    def get_results(self) -> dict[str, Any]:
        """Return the accumulated matrices needed for DPS and GPS computation."""
        if self.n_tokens == 0 or self.xtx is None:
            raise RuntimeError("No tokens processed yet.")

        # Calculate target_variance = sum(||y_i - mean(y)||^2)
        # using the identity: sum((y_i - y_bar)^2) = sum(y_i^2) - (sum(y_i))^2 / N
        mean_y = self.sum_y / self.n_tokens
        # The sum of squared norms of y_i is exactly self.yty
        # The correction term is N * ||mean_y||^2
        correction = self.n_tokens * torch.sum(mean_y**2)
        target_variance = self.yty - correction

        res = {
            "xtx": self.xtx,
            "xty": self.xty,
            "yty": self.yty,
            "target_variance": target_variance,
            "mean_dissim": self.sum_cos_dissim / self.n_tokens,
            "n_tokens": self.n_tokens,
        }

        # If GPS accumulators were updated, calculate its target variance and include them
        if self.yty_gps is not None and float(self.yty_gps.item()) > 0:
            mean_y_gps = self.sum_y_gps / self.n_tokens
            correction_gps = self.n_tokens * torch.sum(mean_y_gps**2)
            target_variance_gps = self.yty_gps - correction_gps
            res.update({
                "xty_gps": self.xty_gps,
                "yty_gps": self.yty_gps,
                "target_variance_gps": target_variance_gps,
            })

        return res
