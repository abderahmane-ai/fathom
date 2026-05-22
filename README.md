# Recurrent Residuals & Attention Residuals for Transformers

This repository implements and compares two novel transformer residual mechanisms designed to overcome the "information dilution" problem in very deep networks.

## Overview

Standard Transformer residual connections ($h = h + y$) can dilute early-layer
signals as depth increases. This project compares two residual mechanisms:

1.  **Recurrent Residuals (RR)**: Treats the depth dimension as a recurrent process. A persistent "working memory" state $m$ flows across layers, updated by gated read/write mechanisms.
2.  **Attention Residuals (AttnRes)**: Based on the Moonshot AI paper (arXiv:2603.15031). Uses softmax attention over previous "blocks" of layers to selectively aggregate information.

RR is implemented as a standard Pre-LN residual addition plus a diagonal gated
memory path. Block AttnRes is implemented as the practical Attention Residuals
baseline from the Moonshot AI paper.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/recurrent-residuals.git
cd recurrent-residuals

# Install dependencies (requires Python 3.10+)
pip install .
```

## Training

We use **PyTorch Lightning** and **Hydra** for experiment management.

```bash
# Train standard residual baseline
python src/train.py model=standard

# Train with Recurrent Residuals (Shared Memory)
python src/train.py model=recurrent_residual

# Train with Block Attention Residuals
python src/train.py model=attnres
```

### Configuration Overrides
You can override any parameter via the CLI:
```bash
python src/train.py model.d_model=512 trainer.precision=bf16-mixed
```

## Testing

Run the unit and integration test suite:
```bash
export PYTHONPATH=$PYTHONPATH:.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

## Methodology

For a deep dive into the mathematics of Recurrent Residuals, see [METHODOLOGY.md](METHODOLOGY.md).

### Key Equations (RR)
- **Read**: $h_l = h_{l-1} + y_l + r_l \odot (g_m \odot \text{RMSNorm}(m_{l-1}))$
- **Update**: $m_l = u_l \odot y_l + (1 - u_l) \odot m_{l-1}$

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{recurrent-residuals2026,
  author = {Abdou Magico},
  title = {Recurrent Residuals for Deep Transformers},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```
