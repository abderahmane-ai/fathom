# Methodology: Residual Stream Dynamics in Deep Transformers

## 1. Introduction

Standard transformers use additive residual connections to ease gradient flow. At extreme depth, these connections accumulate layer outputs without selectivity, causing early-layer signals to be progressively obscured — a problem termed **information dilution**. This document defines the mathematics of **five** residual mechanisms compared in this project and explains the evaluation metrics used to measure information preservation.

### 1.1 The Design Ladder

The five mechanisms are not five independent ideas — they form a **design ladder** of progressively richer approximations of the same underlying operation: *letting a layer reach back into the history of hidden states produced earlier in the depth stream.*

| Rung | Mechanism | History representation | Per-sublayer cost | Inductive bias toward attention |
|---:|---|---|---|---|
| 0 | **Standard** | None — only the immediately previous $h$ | $O(d)$ | None |
| 1 | **RR** (Recurrent Residual) | Single gated memory vector $m \in \mathbb{R}^{d}$ | $O(\text{rank} \cdot d)$ | None — implicit via learned gates |
| 2 | **VEGA** (Vertical EMA Gated Attention) | Multi-head linear-attention state $(S, z)$ of size $n_h \cdot r_h$ | $O(n_h \cdot r_h \cdot d)$ | Linear-attention over depth |
| 3 | **mHC** (Multi-Head Hyper-Connections, Lite variant) | $n$ parallel residual channels with input-dependent pre/post/residual mixing | $O(n^2 \cdot d + n \cdot d \cdot n!)$ per layer | None — orthogonal to history aggregation |
| 4 | **AttnRes** (Block / Full) | Full stack of previous $h$'s, softmax-weighted | $O(B \cdot d)$ per block (block variant) | Full softmax over depth |

The middle three rungs (RR, VEGA, AttnRes) can be read as **three different points on the cost/expressivity frontier of "let later layers re-read earlier representations"** — RR is the cheapest and least expressive, VEGA is the sweet spot (linear-attention-style retrieval at $O(L)$ total cost), and AttnRes is the upper bound ($O(L^2)$). mHC sits orthogonally: it is not a history-aggregation scheme but a **parallel-channel mixing** scheme, included as a recently-published reference baseline.

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

Each sublayer projects the current output $\mathbf{y}$ into the EMA space:
$$\mathbf{K} = \mathbf{W}_K \mathbf{y}, \quad \mathbf{V} = \mathbf{W}_V \mathbf{y},
\quad \mathbf{Q} = \mathbf{W}_Q \mathbf{y}$$

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

$$\mathbf{r} = \sigma(\mathbf{W}_\text{up} (\mathbf{W}_\text{down} \mathbf{y})), \quad \boldsymbol{\delta} =
\sigma(\mathbf{w}_d \odot \mathbf{y} + \mathbf{b}_d)$$

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

where $g = \sigma(\mathbf{W}_g \mathbf{y})$ is a write gate that controls how much of $\mathbf{V}$
enters the state.

### 4.5 Initialization & Regularization

- Decay gates initialized log-linearly in a continuous spectrum from $0.0$ to $4.5$ across
  all channels.
- Variance regularization is added to the training objective to prevent decay spectrum collapse
  to a uniform value: $\mathcal{L}_\text{reg} = -0.01 \cdot \text{Var}(\boldsymbol{\alpha})$.
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

## 5. Multi-Head Hyper-Connections (mHC-Lite) — Rung 3

**References:**
- Bian, Y., Wang, L., et al., "Multi-Head Hyper-Connections" (arXiv:2512.24880, Dec 2025).
- Yang, Z. & Gao, X., "mHC-Lite: Doubly-Stochastic Routing via Permutation Convex Combinations" (arXiv:2601.05732, Jan 2026).

mHC is **not a history-aggregation scheme** — it sits orthogonally to RR / VEGA / AttnRes on the design ladder. Where the other three rungs ask *"how do we let a layer reach back into the history of previous hidden states?"*, mHC asks *"how do we let a single sublayer mix across multiple parallel residual channels?"*

The mechanism: each layer carries $n$ parallel channels of residual state $H_l \in \mathbb{R}^{B \times S \times n \times d}$ instead of a single $h_l \in \mathbb{R}^{B \times S \times d}$. Before and after each sublayer, three **input-dependent** mixing tensors (computed from the input) route information between channels.

### 5.1 Per-Sublayer Equations (mHC-Lite, $n=2$)

Let $\text{RMSNorm}(H) \in \mathbb{R}^{B \times S \times n \times d}$ denote the parameter-free RMSNorm of the channel state, $\alpha_\star$ a learnable scalar temperature, and $W_\star \in \mathbb{R}^{(n \cdot d) \times k}$ a learnable projection (with $k = n$ for $H_\text{pre}, H_\text{post}$ and $k = n!$ for $H_\text{res}$). Then for each sublayer:

$$\hat{H} = \text{RMSNorm}(H_l) \qquad \text{(parameter-free, channel-wise RMSNorm)}$$

$$H_\text{pre}^{(b,s,i)} = \sigma\!\left(\alpha_\text{pre} \cdot \langle \hat{H}^{(b,s)}, W_\text{pre} \rangle + b_\text{pre}^{(i)}\right) \in \mathbb{R}^{n} \qquad \text{(pre-mix weights, sigmoid)}$$

$$H_\text{post}^{(b,s,i)} = 2 \cdot \sigma\!\left(\alpha_\text{post} \cdot \langle \hat{H}^{(b,s)}, W_\text{post} \rangle + b_\text{post}^{(i)}\right) \in \mathbb{R}^{n} \qquad \text{(post-mix weights, $2\times$ sigmoid)}$$

$$H_\text{res}^{(b,s)} = \text{softmax}\!\left(\alpha_\text{res} \cdot \langle \hat{H}^{(b,s)}, W_\text{res} \rangle + b_\text{res}\right) \in \mathbb{R}^{n!} \qquad \text{(residual-mix weights)}$$

$$x_\text{pre} = \sum_{i} H_\text{pre}^{(i)} \cdot H_l^{(i)} \in \mathbb{R}^{d} \qquad \text{(pre-mix: blend channels → sublayer input)}$$

$$y = \mathcal{F}(\text{LayerNorm}(x_\text{pre})) \qquad \text{(standard sublayer output)}$$

$$H_\text{mixed} = \sum_{\pi \in S_n} H_\text{res}^{(\pi)} \cdot \pi(H_l) \in \mathbb{R}^{n \times d} \qquad \text{(residual-mix: doubly-stochastic channel carry-over)}$$

$$H_{l+1} = H_\text{mixed} + y \otimes H_\text{post} \qquad \text{(post-mix: add sublayer output back to channels)}$$

For $n=2$ the only permutations are the identity $\pi_0$ and the swap $\pi_1$, so the residual-mix is the convex combination $H_\text{res}^{(0)} \cdot H + H_\text{res}^{(1)} \cdot \pi_1(H)$. The $2\times$ factor on $H_\text{post}$ is from the paper — it ensures the average y-gain is 1 at init (rather than $1/n$).

### 5.2 Approximate Zero-Start Initialization

mHC's init follows the paper's bias-only protocol: all projection weights start at zero, all temperatures start at $\alpha_\star = 0.01$, and the bias vectors are set to give the desired init behavior:

| Tensor | Shape | Init | Value at init |
|--------|-------|------|---------------|
| $W_\text{pre}, W_\text{post}, W_\text{res}$ | $(n \cdot d) \times k$ | $\mathbf{0}$ | 0 |
| $\alpha_\text{pre}, \alpha_\text{post}, \alpha_\text{res}$ | scalar | $0.01$ | 0.01 |
| $b_\text{pre}, b_\text{post}$ | $(1, n)$ | $+1$ on main channel, $-1$ on shadow | $[+1, -1]$ |
| $b_\text{res}$ | $(1, n!)$ | $0$ on identity, $-8$ on swap | $[0, -8]$ |

Plugging in the bias values at init (with $W=0$, so the projections vanish):

$$H_\text{pre} = \sigma([+1, -1]) \approx [0.731, 0.269] \qquad H_\text{post} = 2 \cdot \sigma([+1, -1]) \approx [1.462, 0.538]$$

$$H_\text{res} = \text{softmax}([0, -8]) \approx [0.9997, 0.0003] \quad \text{so} \quad H_\text{res} \cdot H \approx I_2 \cdot H$$

This gives an **approximate** zero-start (NOT bit-for-bit, unlike the static mHC scheme that this code used to implement):

- Main channel: $H_{l+1}^{(0)} \approx H_l^{(0)} + 1.462 \cdot y$ (close to a standard residual, but the y-gain is 1.462 not 1).
- Shadow channel: $H_{l+1}^{(1)} \approx H_l^{(1)} + 0.538 \cdot y$ (passive carry-over, with a small y-leak).

The strict bit-for-bit zero-start that the **static** mHC scheme provides is **not achievable with the paper's bias-only dynamic init** — it is a property of the simpler linear-mixing version, not the full mHC. The dynamic version trades strictness for the input-dependent routing the paper actually proposes. The at-init values are verified by `tests/test_design_ladder.py::test_mhc_approximate_zero_start_at_init` and `test_mhc_init_contract_at_init`.

### 5.3 Parameter Overhead

Per layer, the added parameters are:

- $W_\text{pre}, W_\text{post} \in \mathbb{R}^{n \cdot d \times n}$: $2 n^2 d$ parameters (for $n=2$: $8d$).
- $W_\text{res} \in \mathbb{R}^{n \cdot d \times n!}$: $n \cdot d \cdot n!$ parameters (for $n=2$: $4d$).
- Three learnable $\alpha_\star$ scalars: 3 parameters.
- Three bias vectors of shape $(1, n)$: $3n$ parameters (for $n=2$: 6).

Total per layer: $2 n^2 d + n \cdot d \cdot n! + 3n + 3$ (for $n=2$: $12d + 9$, i.e. $\sim 0.1\%$ of a 1024-dim model). The $n!$ term is why we only support $n=2$ in this project — for $n=4$ it is $24 \cdot n \cdot d$ and for $n=8$ it is $40320 \cdot n \cdot d$, out of scope. The sHC paper (arXiv:2603.20896) is the polynomial-scaling alternative for larger $n$.

### 5.4 Relation to the Design Ladder

mHC is **orthogonal** to the history-aggregation rungs. It does not let a layer re-read earlier *outputs*; it lets a single sublayer *mix across* $n$ parallel residual streams. The two ideas are composable in principle (one could imagine a "mHC + AttnRes" hybrid where each of the $n$ channels uses a different history scheme), but the implementation in this project keeps them separate so each can be evaluated in isolation.

In the benchmark suite, mHC is the **recently-published reference baseline** — included as the "what does the latest concurrent work propose?" comparison point, with the mHC-Lite ($n=2$) variant chosen to keep parameter overhead negligible against the model size.

---

## 6. Attention Residuals (AttnRes) — Moonshot AI (Rung 4)

**Reference:** Kimi Team / Moonshot AI, "Attention Residuals", arXiv:2603.15031.

AttnRes treats the depth dimension as a sequence and uses a **learned pseudo-query** to softmax-aggregate over previous block states. The practical variant, **BlockAttnRes**, groups layers into blocks to keep memory cost bounded.

### 5.1 BlockAttnRes Equations

Let $B_0, B_1, \ldots, B_{n-1}$ be the states at completed block boundaries, and $B_\text{cur}$ be the current in-block accumulation.

$$\text{values} = \text{stack}([B_0, B_1, \ldots, B_{n-1}, B_\text{cur}]) \in \mathbb{R}^{(n+1) \times d}$$

$$\text{logits} = \frac{\mathbf{q}^\top \text{RMSNorm}(\text{values})}{\sqrt{d_{model}}} \quad (\mathbf{q} \in \mathbb{R}^d \text{ is the pseudo-query})$$

$$\mathbf{w} = \text{softmax}(\text{logits}) \in \mathbb{R}^{n+1}$$

$$\mathbf{h}_\text{new} = \sum_{i} w_i \, \text{values}_i$$

**Zero-start:** $\mathbf{q} = \mathbf{0}$ at init → logits are all zero → uniform weights → output is the mean of all block states. This is a well-defined, stable starting point that degrades gracefully to a mean residual.

### 6.2 FullAttnRes

Keeps the complete history of all sublayer states. Every layer attends to every prior state. Memory cost is $O(2L \times d)$, limiting practical use to small models. Used in this project as a diagnostic reference.

### 6.3 Complexity Comparison

| Variant | Memory per Token | Compute per Sublayer |
|---|---|---|
| BlockAttnRes (block size B) | O(B · d) | O(B · d) |
| FullAttnRes | O(2L · d) | O(L · d) |

### 6.4 Relation to the Design Ladder

AttnRes is the **upper-bound rung** of the history-aggregation axis: it is the only mechanism in the ladder that does exact softmax over the full history. Its init behavior is "uniform mean" rather than "passthrough" — so the first gradient step is meaningfully different from a standard model, and the training-dynamics comparison in `scaling_efficiency` is between a model that starts as an *approximate passthrough* (mHC, see §5.2), a model that starts as a *soft passthrough* (RR/VEGA), and a model that starts as a *uniform mean* (AttnRes).

In the benchmark suite, BlockAttnRes is the **quadratic target** — what the cheaper alternatives are trying to approximate.

---

## 7. Summary Comparison

| Property | Standard | RR | VEGA | mHC | AttnRes (Block) |
|---|---|---|---|---|---|
| **Rung** | 0 | 1 | 2 | 3 | 4 |
| **Mechanism** | Fixed addition | Gated recurrency | Linear-attention EMA | Dynamic parallel-channel mixing | Softmax over blocks |
| **History aggregation** | None | Implicit (gates) | QKV linear-attention | None (orthogonal) | Full softmax |
| **Complexity / sublayer** | O(d) | O(rank · d) | O(n_h · r_h · d) | O(n²·d + n·d·n!) | O(B · d) |
| **Memory / token** | O(d) | O(2d) | O(n_h · r_h²) | O(n · d) | O(B · d) |
| **Init behavior** | N/A | 0.953·h + y | 0.953·h + y | 1.462·y[0] + 0.538·y[1] on H ≈ I (ch 0) | mean(history) |
| **Strict zero-start** | — | soft | soft | no (paper's approximate) | no |
| **New hyperparameters** | None | 0 (bias values) | rank, n_heads, fast/slow | num_channels | block_size |
| **Reference** | — | this work | this work | Bian et al. 2024 | Kimi / Moonshot 2025 |

---

## 8. Evaluation Metrics

### 8.1 Depth Preservation Score (DPS) and Dilution Resistance Index (DRI)

**DPS(k)** measures how linearly accessible layer $k$'s representation is from the final hidden state $\mathbf{s}$. A Ridge regression probe (λ = 1) is fit from $\mathbf{s}$ to each normalized intermediate activation $\mathbf{a}_k$:

$$\text{DPS}(k) = 1 - \frac{\sum_i \|\mathbf{a}_k^{(i)} - (\mathbf{W} \mathbf{s}^{(i)} + \mathbf{b})\|^2}{\sum_i \|\mathbf{a}_k^{(i)} - \bar{\mathbf{a}}_k\|^2}$$

**DRI** averages DPS over the early half of the network (layers 1 to ⌊L/2⌋), summarizing overall resistance to information dilution.

### 8.2 Gradient Preservation Score (GPS) and Gradient Preservation Index (GPI)

**GPS(k)** measures whether the task-relevant direction at layer $k$ is still recoverable from the final state. The implicit early gradient is:

$$\mathbf{g}_k^{(i)} = \mathbf{W}_\text{head}^\top (\text{softmax}(\mathbf{a}_k^{(i)} \mathbf{W}_\text{head}) - \mathbf{y}^{(i)})$$

A Ridge regression probe is fit from $\mathbf{s}$ to $\mathbf{g}_k$, and GPS is the resulting $R^2$ score. **GPI** is the average GPS over the first half of the network.

### 8.3 Interpretation

| DRI | GPI | Perplexity | Interpretation |
|---|---|---|---|
| High | High | Better or equal | Successful preservation |
| High | Low | Better or equal | Healthy abstraction (model discards task-irrelevant detail) |
| High | Low | Worse | Cluttering (model forced to retain irrelevant features) |
| Low | Low | Worse | Classic dilution |
| Low | High | Any | Targeted preservation (raw signal compressed but task direction retained) |

---

## 9. Conclusion

This project compares **five points in the design space of depth-stream residuals**, organized as a design ladder of progressively richer approximations of the same underlying operation:

- **Standard** residuals are the free baseline — passive accumulation, no overhead.
- **RR** (Rung 1) provides $O(\text{rank} \cdot d)$ gated working memory with soft zero-start — the simplest recurrent cell that can in principle learn to summarize depth history.
- **VEGA** (Rung 2) is linear attention run vertically across depth — multi-head linear-attention retrieval over a fixed-size state, with QKV inductive bias and per-head decay horizons.
- **mHC** (Rung 3) is orthogonal to the history-aggregation axis — parallel residual channels with input-dependent pre/post/residual mixing, with an *approximate* zero-start on the main channel ($H_\text{res} \cdot H + 1.462 \cdot y$, not exactly $H + y$ — see §5.2 for why).
- **BlockAttnRes** (Rung 4, Moonshot AI) is the upper bound on the history-aggregation axis — full softmax over block history at $O(B \cdot d)$ cost per sublayer.

The empirical question this project is designed to answer is: **on the history-aggregation axis (RR → VEGA → AttnRes), how much of AttnRes's quality can be recovered at $O(L)$ cost, and is the gap large enough to justify the $O(L^2)$ price?** The `scaling_efficiency` benchmark is the direct measurement; `depth_needle` measures the limit case (single-token retrieval at depth), and `depth_preservation` measures how well each mechanism preserves the *probe accessibility* of intermediate representations.

All four alternatives share a common zero-start protocol: at initialization they each reduce to either an exact or a soft variant of a standard Pre-LN residual, ensuring training stability while allowing gradual specialization. The strictness of the zero-start (paper-faithful approximate passthrough for mHC, 0.953·h for RR/VEGA, uniform mean for AttnRes) is a real design choice that affects the first few hundred training steps and is verified empirically in `tests/test_design_ladder.py`.
