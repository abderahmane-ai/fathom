# Methodology: Residual Stream Dynamics in Deep Transformers

## 1. Introduction

Standard transformers use additive residual connections to ease gradient flow. At extreme depth, these connections accumulate layer outputs without selectivity, causing early-layer signals to be progressively obscured — a problem termed **information dilution**. This document defines the mathematics of **four** residual mechanisms compared in this project and explains the evaluation metrics used to measure information preservation.

### 1.1 The Design Ladder

The mechanisms form a **design ladder** of progressively richer approximations of the same underlying operation: *letting a layer reach back into the history of hidden states produced earlier in the depth stream.*

| Rung | Mechanism | History representation | Per-sublayer cost | Inductive bias toward attention |
|---:|---|---|---|---|
| 0 | **Standard** | None — only the immediately previous $h$ | $O(d)$ | None |
| 1 | **RR** (Recurrent Residual) | Single gated memory vector $m \in \mathbb{R}^{d}$ | $O(\text{rank} \cdot d)$ | None — implicit via learned gates |
| 2 | **VEGA** (Vertical EMA Gated Attention) | Multi-head linear-attention state $(S, z)$ of size $n_h \cdot r_h$ | $O(n_h \cdot r_h \cdot d)$ | Linear-attention over depth |
| 3 | **AttnRes** (Block / Full) | Full stack of previous $h$'s, softmax-weighted | $O(B \cdot d)$ per block (block variant) | Full softmax over depth |

Rungs 1–3 form the **history-aggregation axis**: RR is the cheapest and least expressive, VEGA is the sweet spot (linear-attention-style retrieval at $O(L)$ total cost), and AttnRes is the upper bound ($O(L^2)$ full softmax).

In token-axis language: **VEGA is to AttnRes what RWKV / Gated Linear Attention is to softmax attention** — same query-conditioned retrieval idea, but a closed-form linear recurrence over a fixed-size state instead of an explicit softmax over a growing key set.

The remaining sections derive each mechanism from scratch and state the **init-time equivalence** that every alternative in this ladder is required to satisfy: at initialization, the alternative must reduce to a standard Pre-LN residual, so that a model with the alternative swapped in can be trained with the same hyperparameters as a standard model and the alternative's behaviour can only diverge from standard through learning.

---

## 2. Standard Residuals (Baseline)

In a Pre-LN transformer, each sublayer computes:

$$h_{l+1} = h_l + \mathcal{F}_l(\text{LayerNorm}(h_l))$$

The skip connection is a **fixed, uniform weight** of 1. No gate selects which historical information is relevant. In a 100-layer model, every early representation is smeared across a sum of 200 sublayer outputs.

This is the **zero-start fixed point** that every alternative must reproduce at init — deviations from it at init will compound across the depth stream and either destabilize training or bias the loss landscape before learning has begun.

---

## 3. Recurrent Residuals (RR) — Rung 1

RR replaces uniform accumulation with a **gated depth-wise working memory** $\mathbf{m}$ that flows through the entire layer stack alongside the hidden state $\mathbf{h}$. As the lowest rung on the design ladder, it is the **simplest recurrent cell that can in principle learn to summarize history**: a single linear memory $m$ updated by a gated linear recurrence, with no explicit QKV projection and therefore no built-in inductive bias toward attention.

### 3.1 Per-Sublayer Equations

$$\mathbf{y}_l = \mathcal{F}_l(\text{LayerNorm}(\mathbf{h}_{l-1}))$$

**Read gate** (how much memory to inject):
$$\mathbf{r}_l = \sigma(\mathbf{W}_{r,\text{up}}(\mathbf{W}_{r,\text{down}} \text{RMSNorm}(\mathbf{y}_l)) + \mathbf{p}_r[l])$$

**Damp gate** (how much of the previous hidden state to keep):
$$\mathbf{d}_l = \sigma(\mathbf{W}_{d,\text{up}}(\mathbf{W}_{d,\text{down}} \text{RMSNorm}(\mathbf{y}_l)) + \mathbf{p}_d[l])$$

**Hidden state update**:
$$\mathbf{h}_l = \mathbf{d}_l \odot \mathbf{h}_{l-1} + \mathbf{y}_l + \mathbf{r}_l \odot (\mathbf{g}_m \odot \mathbf{W}_{\text{out}}(\text{RMSNorm}(\mathbf{m}_{l-1})))$$

**Forget gate** (how much of the old memory to retain):
$$\mathbf{f}_l = \sigma(\mathbf{W}_{f,\text{up}}(\mathbf{W}_{f,\text{down}} \text{RMSNorm}(\mathbf{m}_{l-1})) + \mathbf{p}_f[l])$$

**Update gate** (how aggressively to write the new output):
$$\mathbf{u}_l = \sigma(\mathbf{W}_{u,\text{up}}(\mathbf{W}_{u,\text{down}} \text{RMSNorm}(\mathbf{y}_l)) + \mathbf{p}_u[l])$$

**Memory update**:
$$\mathbf{m}_l = \mathbf{f}_l \odot \mathbf{m}_{l-1} + \mathbf{u}_l \odot \tanh(\mathbf{y}_l)$$

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

**At init this gives the soft zero-start** $h_l \approx 0.953 \cdot h_{l-1} + y_l$ — almost the standard residual, with a small uniform attenuation. The cell *writes* into the memory (update gate is not closed) but the memory *read-out* is gated off, so the depth cell has no net effect on $h$ at step 0. This is verified by `tests/test_design_ladder.py::test_rr_zero_start_at_init`.

### 3.3 Parameter Overhead

For a model with $L$ layers and $S = 2L$ sublayers, depth biases add $4 \times S \times d$ parameters.  Memory state size is $2d$ per token (both $\mathbf{h}$ and $\mathbf{m}$ are retained).

### 3.4 Relation to the Design Ladder

RR is the **rank-1 recurrent rung**: it can in principle learn a depth summary, but has no explicit QKV projection, so the kind of summary it can learn is unconstrained by an attention inductive bias. Compared to VEGA, RR is cheaper (state size $2d$ vs. $n_h \cdot r_h$) and more interpretable (a single memory vector you can read directly), at the cost of expressivity. In the benchmark suite, RR is the **first-order baseline** against which VEGA's linear-attention retrieval is measured.

---

## 4. VEGA — Vertical EMA Gated Attention (Rung 2)

VEGA is **linear attention run vertically across depth**. If AttnRes is `softmax-attention over depth` at $O(L^2)$ cost, VEGA is `linear-attention over depth` at $O(L)$ cost — the depth-axis analog of how RWKV / GLA / Linear Transformers approximate softmax attention in the token axis. The state size is conditional on the head rank: it uses a **Vector State**
$\mathbf{S} \in \mathbb{R}^{n_{\text{heads}} \times r_{\text{head}}}$ ($O(r)$ memory) if
$r_{\text{head}} \le 64$ to eliminate cross-channel mixing overhead at small ranks, and falls
back to a **Matrix State** $\mathbf{S} \in \mathbb{R}^{n_{\text{heads}} \times r_{\text{head}}
\times r_{\text{head}}}$ ($O(r^2)$ memory) for $r_{\text{head}} \ge 128$.

### 4.1 Projections

Each sublayer projects the current output $\mathbf{y}$ into the EMA space (after parameter-free RMSNorm):
$$\mathbf{K} = \mathbf{W}_K \text{RMSNorm}(\mathbf{y}), \quad \mathbf{V} = \mathbf{W}_V \text{RMSNorm}(\mathbf{y}),
\quad \mathbf{Q} = \mathbf{W}_Q \text{RMSNorm}(\mathbf{y})$$

Per-sublayer depth query bias is added (key depth bias is omitted):
$$\mathbf{Q}_{dep} = \mathbf{Q} + b_Q[pos], \quad \mathbf{K}_{dep} = \mathbf{K}$$

To ensure numerical stability and prevent state growth, the query and key vectors are
normalized and scaled before the positive feature map:
$$\mathbf{K}_{dep} \leftarrow \text{RMSNorm}(\mathbf{K}_{dep}) \times \frac{1}{\sqrt{r_{head}}}$$
$$\mathbf{Q}_{dep} \leftarrow \text{RMSNorm}(\mathbf{Q}_{dep}) \times \frac{1}{\sqrt{r_{head}}}$$

The positive feature map $\phi(\mathbf{x}) = \text{ELU}(\mathbf{x}) + 1$ ensures state positivity.

### 4.2 State Retrieval

Linear-attention retrieval from the previous state $\mathbf{S}_{prev}$:

- **Vector State** ($r_{head} \le 64$):
  $$c = \frac{\phi(\mathbf{Q}_{dep}) \odot \mathbf{S}_{prev}}{(\phi(\mathbf{Q}_{dep}) \odot
  \mathbf{z}_{prev}).\text{sum}(-1) + \varepsilon}$$

- **Matrix State** ($r_{head} \ge 128$):
  $$c = \frac{\phi(\mathbf{Q}_{dep})^\top \mathbf{S}_{prev}}{\phi(\mathbf{Q}_{dep})^\top
  \mathbf{z}_{prev} + \varepsilon}$$

The retrieval $c$ is normalized and projected through a single output layer:
$$c_\text{out} = \mathbf{W}_\text{out}(\text{RMSNorm}(c))$$

### 4.3 Hidden State Update

Using a single low-rank read gate $\mathbf{r}$ and element-wise damp gate $\boldsymbol{\delta}$:

$$\mathbf{r} = \sigma(\mathbf{W}_\text{up} (\mathbf{W}_\text{down} \text{RMSNorm}(\mathbf{y}))), \quad \boldsymbol{\delta} =
\sigma(\mathbf{w}_d \odot \text{RMSNorm}(\mathbf{y}) + \mathbf{b}_d)$$

$$\mathbf{h}_\text{new} = \boldsymbol{\delta} \odot \mathbf{h}_\text{prev} +
\mathbf{y} + \mathbf{r} \odot c_\text{out}$$

### 4.4 EMA State Update

$$\boldsymbol{\alpha} = \sigma(\text{decay}[pos]) \quad (\text{per-head, per-rank decay gates})$$

- **Vector State** ($r_{head} \le 64$):
  $$\mathbf{S}_\text{new} = \boldsymbol{\alpha} \odot \mathbf{S}_\text{prev} +
  \phi(\mathbf{K}_{dep}) \odot (g \odot \mathbf{V})$$

- **Matrix State** ($r_{head} \ge 128$):
  $$\mathbf{S}_\text{new} = \boldsymbol{\alpha} \odot \mathbf{S}_\text{prev} +
  \phi(\mathbf{K}_{dep})^\top \otimes (g \odot \mathbf{V})$$

For both states, the normalization update is:
$$\mathbf{z}_\text{new} = \boldsymbol{\alpha} \odot \mathbf{z}_\text{prev} + \phi(\mathbf{K}_{dep})$$

where $g = \sigma(\mathbf{W}_g \text{RMSNorm}(\mathbf{y}))$ is a write gate that controls how much of $\mathbf{V}$
enters the state.

### 4.5 Initialization & Regularization

- Decay gates initialized log-linearly in a split spectrum: fast heads cover $0.0$–$1.2$,
  slow heads cover $2.0$–$4.5$ (with a gap between the two groups by design).
- Variance regularization is added to the training objective to prevent decay spectrum collapse
  to a uniform value: $\mathcal{L}_\text{reg} = -0.001 \cdot \text{Var}(\boldsymbol{\alpha})$.
- Output projection $\mathbf{W}_\text{out}$ is zero-initialized → at init the retrieved
  $c_\text{out}$ is zero, so VEGA produces $h_l \approx 0.953 \cdot h_{l-1} + y_l$ — the same
  soft zero-start as RR (verified by `tests/test_design_ladder.py::test_vega_zero_start_at_init`).
- Key, value, query projections initialized orthogonally for conditioning.

### 4.6 Relation to the Design Ladder

VEGA is the **multi-head linear-attention rung**: it has the full QKV projection + φ feature map + EMA recurrence structure of a linear attention cell, just with the iteration axis swapped from tokens to depth. Compared to RR, VEGA is more expressive (it can learn *which* earlier-layer outputs are relevant to the current layer's query, not just accumulate them); compared to AttnRes, it is much cheaper ($O(L)$ instead of $O(L^2)$) but cannot do exact softmax selection over the full history.

In the benchmark suite, VEGA is the **central alternative** — the question "does linear-attention-style depth retrieval match the expressivity of full softmax attention over depth, at $O(L)$ cost?" is what the Pareto frontier in `scaling_efficiency` is designed to answer.

### 4.7 Implementation Optimizations

To maximize training throughput and prevent GPU kernel launch latency bottlenecks:
- **Projection Fusion**: $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ projections are fused into a
  single combined projection layer:
  $$\begin{bmatrix} \mathbf{Q} \\ \mathbf{K} \\ \mathbf{V} \end{bmatrix} = \mathbf{W}_{qkv} \mathbf{y}$$
    and chunked back to independent streams in memory.
- **Kernel Fusion**: Using `torch.compile` allows PyTorch to dynamically compile and fuse element-wise
  operations (such as Sigmoid, ELU, and scaling updates) into a single combined CUDA kernel per
  sublayer, bypassing dispatch latency.

---

## 5. Attention Residuals (AttnRes) — Moonshot AI (Rung 3)

**Reference:** Kimi Team / Moonshot AI, "Attention Residuals", arXiv:2603.15031.

AttnRes treats the depth dimension as a sequence and uses a **learned pseudo-query** to softmax-aggregate over previous block states. The practical variant, **BlockAttnRes**, groups layers into blocks to keep memory cost bounded.

### 5.1 BlockAttnRes Equations

Let $B_0, B_1, \ldots, B_{n-1}$ be the states at completed block boundaries, and $B_\text{cur}$ be the current in-block accumulation.

$$\text{values} = \text{stack}([B_0, B_1, \ldots, B_{n-1}, B_\text{cur}]) \in \mathbb{R}^{(n+1) \times d}$$

$$\text{logits} = \mathbf{q}^\top \text{RMSNorm}(\text{values}) \quad (\mathbf{q} \in \mathbb{R}^d \text{ is the pseudo-query, no }\sqrt{d}\text{ scaling — paper's formula})$$

$$\mathbf{w} = \text{softmax}(\text{logits}) \in \mathbb{R}^{n+1}$$

$$\mathbf{h}_\text{new} = \sum_{i} w_i \, \text{values}_i$$

**Zero-start:** $\mathbf{q} = \mathbf{0}$ at init → logits are all zero → uniform weights → output is the mean of all block states. This is a well-defined, stable starting point that degrades gracefully to a mean residual.

### 5.2 FullAttnRes

Keeps the complete history of all sublayer states. Every layer attends to every prior state. Memory cost is $O(2L \times d)$, limiting practical use to small models. Used in this project as a diagnostic reference.

### 5.3 Complexity Comparison

| Variant | Memory per Token | Compute per Sublayer |
|---|---|---|
| BlockAttnRes (block size $B$) | $O(B \cdot d)$ | $O(B \cdot d)$ |
| FullAttnRes | $O(2L \cdot d)$ | $O(L \cdot d)$ |

### 5.4 Relation to the Design Ladder

AttnRes is the **upper-bound rung** of the history-aggregation axis: it is the only mechanism in the ladder that does exact softmax over the full history. Its init behavior is "uniform mean" rather than "passthrough" — so the first gradient step is meaningfully different from a standard model, and the training-dynamics comparison is between models that start as a *soft passthrough* (RR/VEGA) and a model that starts as a *uniform mean* (AttnRes).

In the benchmark suite, BlockAttnRes is the **quadratic target** — what the cheaper alternatives are trying to approximate.

---

## 6. Summary Comparison

| Property | Standard | RR | VEGA | AttnRes (Block) |
|---|---|---|---|---|
| **Rung** | $0$ | $1$ | $2$ | $3$ |
| **Mechanism** | Fixed addition | Gated recurrence | Linear-attention EMA | Softmax over blocks |
| **History aggregation** | None | Implicit (gates) | QKV linear-attention | Full softmax |
| **Complexity / sublayer** | $O(d)$ | $O(r \cdot d)$ | $O(n_h \, r_h \, d)$ | $O(B \cdot d)$ |
| **Memory / token** | $O(d)$ | $O(2d)$ | $O(n_h \, r_h)$ | $O(B \cdot d)$ |
| **Init behavior** | $h + y$ (exact) | $\approx\! 0.953h + y$ | $\approx\! 0.953h + y$ | $\mathrm{mean}(\mathrm{history})$ |
| **Strict zero-start** | exact | soft | soft | none |
| **New hyperparameters** | — | bias values | $r_h$, $n_h$, decay range | block size $B$ |
| **Reference** | — | this work | this work | Moonshot AI (2025) |

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

This project compares **four points in the design space of depth-stream residuals**, organized as a design ladder of progressively richer approximations of the same underlying operation:

- **Standard** residuals are the free baseline — passive accumulation, no overhead.
- **RR** (Rung 1) provides $O(\text{rank} \cdot d)$ gated working memory with soft zero-start — the simplest recurrent cell that can in principle learn to summarize depth history.
- **VEGA** (Rung 2) is linear attention run vertically across depth — multi-head linear-attention retrieval over a fixed-size state, with QKV inductive bias and per-head decay horizons.
- **BlockAttnRes** (Rung 3, Moonshot AI) is the upper bound on the history-aggregation axis — full softmax over block history at $O(B \cdot d)$ cost per sublayer.

The empirical question this project is designed to answer is: **on the history-aggregation axis (RR → VEGA → AttnRes), how much of AttnRes's quality can be recovered at $O(L)$ cost, and is the gap large enough to justify the $O(L^2)$ price?** The `scaling_efficiency` benchmark is the direct measurement; `depth_needle` measures the limit case (single-token retrieval at depth), and `depth_preservation` measures how well each mechanism preserves the *probe accessibility* of intermediate representations.

All alternatives share a common zero-start protocol: at initialization they each reduce to either an exact or a soft variant of a standard Pre-LN residual, ensuring training stability while allowing gradual specialization. The strictness of the zero-start (0.953·h for RR/VEGA, uniform mean for AttnRes) is a real design choice that affects the first few hundred training steps and is verified empirically in `tests/test_design_ladder.py`.
