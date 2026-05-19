# Recurrent Residuals (RR)

A PyTorch implementation of **Recurrent Residuals** and **Block Attention Residuals** for deep Transformer language models. This project explores mechanisms to solve the "information dilution" problem in standard residual connections without the $O(L)$ complexity of full depth-wise attention.

## 🚀 Overview

Standard Transformers accumulate information via simple addition, which can bury early-layer signals as depth increases. This repository implements:

1.  **Standard Residuals**: The classic Pre-LN baseline.
2.  **Recurrent Residuals (RR)**: A gated "Working Memory" mechanism inspired by RNNs, using **DeepNorm** for numerical stability at extreme depths (1,000+ layers).
3.  **Block Attention Residuals (AttnRes)**: A block-wise attention mechanism that allows layers to attend to previous block representations.

## 🛠️ Installation

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## 📈 Training

The project uses [Hydra](https://hydra.cc/) for configuration and [PyTorch Lightning](https://lightning.ai/) for training.

```bash
# Train standard baseline
python src/train.py model=standard

# Train Recurrent Residual model
python src/train.py model=recurrent_residual

# Train Block-AttnRes model
python src/train.py model=attnres
```

### Key Overrides
- `trainer.optimizer.lr=1e-4`: Set learning rate.
- `trainer.devices=2 trainer.strategy=ddp`: Multi-GPU training.
- `model.num_layers=24`: Change model depth.

## 🧪 Testing

```bash
export PYTHONPATH=$PYTHONPATH:.
pytest
```

## 📖 Methodology

For a detailed explanation of the math and design philosophy behind Recurrent Residuals, see [methodology.md](./methodology.md).

## 🗂️ Project Structure

- `src/modules/`: Core architectural components.
  - `recurrent_residual.py`: Gated memory cell logic.
  - `transformer.py`: Decoder-only orchestrator.
  - `attention.py`: Flash-attention enabled self-attention.
- `src/train.py`: Training entry point.
- `src/data.py`: WikiText-103 data pipeline.
- `conf/`: Hydra configuration files.
