import torch
import torch.nn as nn

from benchmarks.common.dps_extractor import DPSEvaluator
from benchmarks.common.metrics import calculate_dri, calculate_gpi, compute_dps_closed_form


class DummyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        return self.linear(x)


class DummyModel(nn.Module):
    def __init__(self, d, num_layers=3, vocab_size=10):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(d) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def test_compute_dps_closed_form():
    """Test that the streaming closed-form matches the naive full-matrix implementation."""
    N = 1000
    d = 16
    lambda_val = 1.0

    # Generate random data
    X = torch.randn(N, d)  # source (final layer)
    Y = torch.randn(N, d)  # target (early layer, already layer-normed)

    # 1. Naive Full-Matrix implementation
    X_tilde = torch.cat([X, torch.ones(N, 1)], dim=1)
    reg = torch.eye(d + 1)
    reg[-1, -1] = 0.0
    w_tilde = torch.linalg.solve(X_tilde.T @ X_tilde + lambda_val * reg, X_tilde.T @ Y)
    Y_pred = X_tilde @ w_tilde

    rss = torch.sum((Y - Y_pred) ** 2)
    mean_Y = Y.mean(dim=0)
    tss = torch.sum((Y - mean_Y) ** 2)
    expected_dps = 1.0 - (rss / tss)

    # 2. Streaming / Accumulated variables
    xtx = X_tilde.T @ X_tilde
    xty = X_tilde.T @ Y
    yty = torch.sum(Y**2)
    target_variance = tss

    actual_dps = compute_dps_closed_form(xtx, xty, yty, target_variance, lambda_val)

    assert torch.isclose(torch.tensor(actual_dps), torch.tensor(expected_dps.item()), atol=1e-4)


def test_dps_extractor():
    """Test that the DPSEvaluator correctly accumulates streaming covariance vs full concat."""
    N = 100
    batch_size = 20
    d = 16
    layer_idx = 0

    model = DummyModel(d=d)
    evaluator = DPSEvaluator(model, layer_idx=layer_idx, final_norm_name="norm")

    all_targets = []
    all_sources = []

    # Run batches
    for _ in range(0, N, batch_size):
        x = torch.randn(batch_size, 5, d)  # (batch, seq, d)

        # We need to manually capture what the hook would capture for the naive check
        # For naive, we can just run it
        model(x)

        # Capture from the evaluator's temporary storage BEFORE process_batch clears it
        # pyrefly: ignore [missing-attribute]
        target = evaluator._current_target.clone().reshape(-1, d)
        # pyrefly: ignore [missing-attribute]
        source = evaluator._current_source.clone().reshape(-1, d)

        # Manual LayerNorm for naive target
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, unbiased=False, keepdim=True)
        y = (target - mean) / torch.sqrt(var + 1e-5)

        all_targets.append(y)
        all_sources.append(source)

        evaluator.process_batch()

    res = evaluator.get_results()
    evaluator.remove_hooks()

    # Naive Concat
    Y_full = torch.cat(all_targets, dim=0)
    X_full = torch.cat(all_sources, dim=0)
    X_tilde_full = torch.cat([X_full, torch.ones(X_full.size(0), 1)], dim=1)

    expected_xtx = X_tilde_full.T @ X_tilde_full
    expected_xty = X_tilde_full.T @ Y_full
    expected_yty = torch.sum(Y_full**2)
    expected_mean_y = Y_full.mean(dim=0)
    expected_variance = torch.sum((Y_full - expected_mean_y) ** 2)

    assert torch.allclose(res["xtx"], expected_xtx, atol=1e-4)
    assert torch.allclose(res["xty"], expected_xty, atol=1e-4)
    assert torch.allclose(res["yty"], expected_yty, atol=1e-4)
    assert torch.allclose(res["target_variance"], expected_variance, atol=1e-3)


def test_calculate_dri():
    # 5 layers (L=6) => first half is floor(6/2) = 3 layers.
    # Average of first 3 scores.
    scores = [0.8, 0.6, 0.4, 0.2, 0.1]
    expected_dri = (0.8 + 0.6 + 0.4) / 3.0
    actual_dri = calculate_dri(scores)
    assert abs(actual_dri - expected_dri) < 1e-6


def test_calculate_gpi():
    scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    expected_gpi = (0.9 + 0.7) / 2.0  # L=5 => floor(5/2) = 2 layers
    actual_gpi = calculate_gpi(scores[:3])  # pass 3 scores, so L=4, floor(4/2)=2
    assert abs(actual_gpi - expected_gpi) < 1e-6


def test_gps_extractor():
    """Test that the DPSEvaluator correctly accumulates streaming covariance for GPS vs naive."""
    N = 100
    batch_size = 20
    d = 16
    vocab_size = 10
    layer_idx = 0

    model = DummyModel(d=d, vocab_size=vocab_size)
    evaluator = DPSEvaluator(model, layer_idx=layer_idx, final_norm_name="norm")

    all_gps_targets = []
    all_sources = []

    # Run batches
    for _ in range(0, N, batch_size):
        x = torch.randn(batch_size, 5, d)  # (batch, seq, d)
        targets = torch.randint(0, vocab_size, (batch_size, 5))

        model(x)

        # Retrieve early activation a_k and target labels to compute naive g_k
        # pyrefly: ignore [missing-attribute]
        target_act = evaluator._current_target.clone().reshape(-1, d)
        # pyrefly: ignore [missing-attribute]
        source = evaluator._current_source.clone().reshape(-1, d)

        # Manual LayerNorm for target to get y
        mean = target_act.mean(dim=-1, keepdim=True)
        var = target_act.var(dim=-1, unbiased=False, keepdim=True)
        y = (target_act - mean) / torch.sqrt(var + 1e-5)

        # Naive computation of implicit early gradient
        with torch.no_grad():
            early_logits = model.lm_head(y)
            probs = torch.softmax(early_logits, dim=-1)
            targets_flat = targets.reshape(-1)
            e = probs.clone()
            e[torch.arange(batch_size * 5), targets_flat] -= 1.0
            g = e @ model.lm_head.weight

        all_gps_targets.append(g)
        all_sources.append(source)

        evaluator.process_batch(targets=targets)

    res = evaluator.get_results()
    evaluator.remove_hooks()

    # Naive Concat
    G_full = torch.cat(all_gps_targets, dim=0)
    X_full = torch.cat(all_sources, dim=0)
    X_tilde_full = torch.cat([X_full, torch.ones(X_full.size(0), 1)], dim=1)

    expected_xtx = X_tilde_full.T @ X_tilde_full
    expected_xty_gps = X_tilde_full.T @ G_full
    expected_yty_gps = torch.sum(G_full**2)
    expected_mean_g = G_full.mean(dim=0)
    expected_variance_gps = torch.sum((G_full - expected_mean_g) ** 2)

    assert torch.allclose(res["xtx"], expected_xtx, atol=1e-4)
    assert torch.allclose(res["xty_gps"], expected_xty_gps, atol=1e-4)
    assert torch.allclose(res["yty_gps"], expected_yty_gps, atol=1e-4)
    assert torch.allclose(res["target_variance_gps"], expected_variance_gps, atol=1e-3)

