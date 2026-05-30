# Methodology: Residual Stream Dynamics in Deep Transformers

## 1. Introduction

Standard transformers use additive residual connections to ease gradient flow. At extreme depth, these connections accumulate layer outputs without selectivity, causing early-layer signals to be progressively obscured — a problem termed **information dilution**. This document defines the mathematics of four residual mechanisms compared in this project and explains the evaluation metrics used to measure information preservation.

---

## 2. Standard Residuals (Baseline)

In a Pre-LN transformer, each sublayer computes:

$$h_{l+1} = h_l + \mathcal{F}_l(\text{LayerNorm}(h_l))$$

The skip connection is a **fixed, uniform weight** of 1. No gate selects which historical information is relevant. In a 100-layer model, every early representation is smeared across a sum of 200 sublayer outputs.

---

## 3. Recurrent Residuals (RR)

RR replaces uniform accumulation with a **gated depth-wise working memory** $\mathbf{m}$ that flows through the entire layer stack alongside the hidden state $\mathbf{h}$.

### 3.1 Per-Sublayer Equations

$$\mathbf{y}_l = \mathcal{F}_l(\text{LayerNorm}(\mathbf{h}_{l-1}))$$

**Read gate** (how much memory to inject):
$$\mathbf{r}_l = \sigma(\mathbf{W}_r \mathbf{y}_l + \mathbf{p}_r[l])$$

**Damp gate** (how much of the previous hidden state to keep):
$$\mathbf{d}_l = \sigma(\mathbf{W}_d \mathbf{y}_l + \mathbf{p}_d[l])$$

**Hidden state update**:
$$\mathbf{h}_l = \mathbf{d}_l \odot \mathbf{h}_{l-1} + \mathbf{y}_l + \mathbf{r}_l \odot (\mathbf{g}_m \odot \text{RMSNorm}(\mathbf{m}_{l-1}))$$

**Forget gate** (how much of the old memory to retain):
$$\mathbf{f}_l = \sigma(\mathbf{W}_f \text{RMSNorm}(\mathbf{m}_{l-1}) + \mathbf{p}_f[l])$$

**Update gate** (how aggressively to write the new output):
$$\mathbf{u}_l = \sigma(\mathbf{W}_u \mathbf{y}_l + \mathbf{p}_u[l])$$

**Memory update**:
$$\mathbf{m}_l = \mathbf{f}_l \odot \mathbf{m}_{l-1} + \mathbf{u}_l \odot \mathbf{y}_l$$

All gate weights $\mathbf{W}_*$ use a low-rank factorization ($d \to \text{rank} \to d$) and per-sublayer depth position biases $\mathbf{p}_*[l]$.

### 3.2 Zero-Start Initialization

At the start of training the cell behaves exactly like a standard Pre-LN transformer:

| Parameter | Initial Value | Effect |
|---|---|---|
| $\mathbf{g}_m$ | $0$ | Memory injection term is zero |
| bias($\mathbf{r}$) | $-3$ | Read gate $\approx 0.047$ (nearly closed) |
| bias($\mathbf{d}$) | $+3$ | Damp gate $\approx 0.953$ ($h_{prev}$ mostly kept) |
| bias($\mathbf{f}$) | $+3$ | Forget gate $\approx 0.953$ (retentive) |
| bias($\mathbf{u}$) | $-2$ | Update gate $\approx 0.119$ (conservative write) |

As training progresses, $\mathbf{g}_m$ departs from zero and the gates specialise.

### 3.3 Parameter Overhead

For a model with $L$ layers and $S = 2L$ sublayers, depth biases add $4 \times S \times d$ parameters.  Memory state size is $2d$ per token (both $\mathbf{h}$ and $\mathbf{m}$ are retained).

---

## 4. VEGA — Vertical EMA Gated Attention

VEGA maintains a **linear-attention EMA state** $\mathbf{S} \in \mathbb{R}^{n_{\text{heads}} \times r \times r}$ per token that accumulates key-value outer products across depth, with **separate fast and slow head groups** for multi-scale temporal coverage.

### 4.1 Projections

Each sublayer projects the current output $\mathbf{y}$ into the EMA space:
$$\mathbf{K} = \mathbf{W}_K \mathbf{y}, \quad \mathbf{V} = \mathbf{W}_V \mathbf{y}, \quad \mathbf{Q} = \mathbf{W}_Q \mathbf{y}$$

Depth-position biases are added before the feature map:
$$\mathbf{K}_{dep} = \mathbf{K} + b_K[pos], \quad \mathbf{Q}_{dep} = \mathbf{Q} + b_Q[pos]$$

The positive feature map $\phi(\mathbf{x}) = \text{ELU}(\mathbf{x}) + 1$ ensures state positivity.

### 4.2 State Retrieval

Linear-attention retrieval from the previous state $\mathbf{S}_{prev}$:

$$c = \frac{\phi(\mathbf{Q}_{dep})^\top \mathbf{S}_{prev}}{\phi(\mathbf{Q}_{dep})^\top \mathbf{z}_{prev} + \varepsilon}$$

The retrieval $c$ is split into fast and slow halves by head group, then projected through separate output layers:

$$c_\text{out\_fast} = \text{out\_fast}(\text{RMSNorm}(c_\text{fast}))$$
$$c_\text{out\_slow} = \text{out\_slow}(\text{RMSNorm}(c_\text{slow}))$$

### 4.3 Hidden State Update

Separate read gates per timescale — this is the key design distinction:

$$\mathbf{r}_\text{fast} = \sigma(\mathbf{W}_\text{rf} \mathbf{y}), \quad \mathbf{r}_\text{slow} = \sigma(\mathbf{W}_\text{rs} \mathbf{y}), \quad \boldsymbol{\delta} = \sigma(\mathbf{w}_d \odot \mathbf{y} + \mathbf{b}_d)$$

$$\mathbf{h}_\text{new} = \boldsymbol{\delta} \odot \mathbf{h}_\text{prev} + \mathbf{y} + \mathbf{r}_\text{fast} \odot c_\text{out\_fast} + \mathbf{r}_\text{slow} \odot c_\text{out\_slow}$$

### 4.4 EMA State Update

$$\boldsymbol{\alpha} = \sigma(\text{decay}[pos]) \quad (\text{per-head, per-rank decay gates})$$

$$\mathbf{S}_\text{new} = \boldsymbol{\alpha} \odot \mathbf{S}_\text{prev} + \phi(\mathbf{K}_{dep})^\top \otimes (g \odot \mathbf{V})$$
$$\mathbf{z}_\text{new} = \boldsymbol{\alpha} \odot \mathbf{z}_\text{prev} + \phi(\mathbf{K}_{dep})$$

where $g = \sigma(\mathbf{W}_g \mathbf{y})$ is a write gate that controls how much of $\mathbf{V}$ enters the state.

### 4.5 Initialization

- Decay gates initialized log-linearly: fast heads cover depth horizons $[1, \text{rank}]$, slow heads cover $[\text{rank}, 2L]$.
- Output projections $\mathbf{W}_\text{out\_fast}$, $\mathbf{W}_\text{out\_slow}$ zero-initialized → zero-start compliant.
- Key, value, query projections initialized orthogonally for conditioning.

---

## 5. Attention Residuals (AttnRes) — Moonshot AI

**Reference:** Kimi Team / Moonshot AI, "Attention Residuals", arXiv:2603.15031.

AttnRes treats the depth dimension as a sequence and uses a **learned pseudo-query** to softmax-aggregate over previous block states. The practical variant, **BlockAttnRes**, groups layers into blocks to keep memory cost bounded.

### 5.1 BlockAttnRes Equations

Let $B_0, B_1, \ldots, B_{n-1}$ be the states at completed block boundaries, and $B_\text{cur}$ be the current in-block accumulation.

$$\text{values} = \text{stack}([B_0, B_1, \ldots, B_{n-1}, B_\text{cur}]) \in \mathbb{R}^{(n+1) \times d}$$

$$\text{logits} = \mathbf{q}^\top \text{RMSNorm}(\text{values}) \quad (\mathbf{q} \in \mathbb{R}^d \text{ is the pseudo-query})$$

$$\mathbf{w} = \text{softmax}(\text{logits}) \in \mathbb{R}^{n+1}$$

$$\mathbf{h}_\text{new} = \sum_{i} w_i \, \text{values}_i$$

**Zero-start:** $\mathbf{q} = \mathbf{0}$ at init → logits are all zero → uniform weights → output is the mean of all block states. This is a well-defined, stable starting point that degrades gracefully to a mean residual.

### 5.2 FullAttnRes

Keeps the complete history of all sublayer states. Every layer attends to every prior state. Memory cost is $O(2L \times d)$, limiting practical use to small models. Used in this project as a diagnostic reference.

### 5.3 Complexity Comparison

| Variant | Memory per Token | Compute per Sublayer |
|---|---|---|
| BlockAttnRes (block size B) | O(B · d) | O(B · d) |
| FullAttnRes | O(2L · d) | O(L · d) |

---

## 6. Summary Comparison

| Property | Standard | RR | VEGA | AttnRes (Block) |
|---|---|---|---|---|
| **Mechanism** | Fixed addition | Gated recurrency | Multi-scale EMA | Softmax over blocks |
| **Complexity / sublayer** | O(d) | O(rank · d) | O(rank · d) | O(B · d) |
| **Memory / token** | O(d) | O(2d) | O(n_heads · rank²) | O(B · d) |
| **Softmax over depth** | No | No | No | Yes |
| **New hyperparameters** | None | 0 (bias values) | rank, n_heads | block_size |
| **Zero-start compliant** | N/A | Yes | Yes | Yes |

---

## 7. Evaluation Metrics

### 7.1 Depth Preservation Score (DPS) and Dilution Resistance Index (DRI)

**DPS(k)** measures how linearly accessible layer $k$'s representation is from the final hidden state $\mathbf{s}$. A Ridge regression probe (λ = 1) is fit from $\mathbf{s}$ to each normalized intermediate activation $\mathbf{a}_k$:

$$\text{DPS}(k) = 1 - \frac{\sum_i \|\mathbf{a}_k^{(i)} - (\mathbf{W} \mathbf{s}^{(i)} + \mathbf{b})\|^2}{\sum_i \|\mathbf{a}_k^{(i)} - \bar{\mathbf{a}}_k\|^2}$$

**DRI** averages DPS over the early half of the network (layers 1 to ⌊L/2⌋), summarizing overall resistance to information dilution.

### 7.2 Gradient Preservation Score (GPS) and Gradient Preservation Index (GPI)

**GPS(k)** measures whether the task-relevant direction at layer $k$ is still recoverable from the final state. The implicit early gradient is:

$$\mathbf{g}_k^{(i)} = \mathbf{W}_\text{head}^\top (\text{softmax}(\mathbf{a}_k^{(i)} \mathbf{W}_\text{head}) - \mathbf{y}^{(i)})$$

A Ridge regression probe is fit from $\mathbf{s}$ to $\mathbf{g}_k$, and GPS is the resulting $R^2$ score. **GPI** is the average GPS over the first half of the network.

### 7.3 Interpretation

| DRI | GPI | Perplexity | Interpretation |
|---|---|---|---|
| High | High | Better or equal | Successful preservation |
| High | Low | Better or equal | Healthy abstraction (model discards task-irrelevant detail) |
| High | Low | Worse | Cluttering (model forced to retain irrelevant features) |
| Low | Low | Worse | Classic dilution |
| Low | High | Any | Targeted preservation (raw signal compressed but task direction retained) |

---

## 8. Conclusion

This project compares four points in the design space of depth-stream residuals:

- **Standard** residuals are the free baseline — passive accumulation, no overhead.
- **RR** provides $O(\text{rank} \cdot d)$ gated working memory with zero-start compliant initialization and provably bounded norms.
- **VEGA** provides multi-scale EMA depth memory; fast/slow head partitioning lets the model separately track local and long-range depth context.
- **BlockAttnRes** (Moonshot AI) provides softmax-based selective retrieval over block history at $O(B \cdot d)$ cost per sublayer.

All three alternatives share a common zero-start protocol: at initialization they reduce to standard Pre-LN transformers, ensuring training stability while allowing gradual specialization.
