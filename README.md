# Recurrent Residuals & Attention Residuals for Transformers

This repository implements and compares two novel transformer residual mechanisms designed to overcome the "information dilution" problem in very deep networks.

## 🚀 Overview

Standard Transformer residual connections ($h = h + y$) lead to signal dilution as depth increases. This project provides two state-of-the-art solutions:

1.  **Recurrent Residuals (RR)**: Treats the depth dimension as a recurrent process. A persistent "working memory" state $m$ flows across layers, updated by gated read/write mechanisms.
2.  **Attention Residuals (AttnRes)**: Based on the Moonshot AI paper (arXiv:2603.15031). Uses softmax attention over previous "blocks" of layers to selectively aggregate information.

Both mechanisms are integrated with **DeepNorm** scaling to enable stable training of 1,000+ layer architectures.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/recurrent-residuals.git
cd recurrent-residuals

# Install dependencies (requires Python 3.10+)
pip install .
```

## 📈 Training

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

## 🧪 Testing

Run the unit and integration test suite:
```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

## 📖 Methodology

For a deep dive into the mathematics of Recurrent Residuals, see [methodology.md](methodology.md).

### Key Equations (RR)
- **Injection**: $h_{new} = \text{LayerNorm}(\alpha \cdot h_{prev} + y + r \odot W_m \text{RMSNorm}(m))$
- **Update**: $m_{new} = \alpha_{gate} \odot y + (1 - \alpha_{gate}) \odot m$

## 📝 Citation

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
