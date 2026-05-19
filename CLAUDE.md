We are implementing a novel transformer residual mechanism called Recurrent Residuals. 
All code must:
1. Use PyTorch, with idiomatic nn.Module design.
2. Be type‑annotated (e.g., torch.Tensor).
3. Avoid any hardcoded hyperparameters; use a central config.yaml managed by Hydra.
4. Include a dedicated config for the Recurrent Residual module (gates init biases, etc.).
5. Support mixed precision (torch.cuda.amp) and DistributedDataParallel.
6. Use PyTorch Lightning for training orchestration, with WandB logging.
7. Separate modules into src/modules/, training logic into src/train.py, and data into src/data.py.
8. Write unit tests with pytest for every non‑trivial component, using torch.testing.assert_close.
9. Every function must have a docstring describing its purpose, arguments, return values, and any preconditions.
10. Memory and compute efficiency are critical – prefer vectorized operations, no unnecessary loops.
