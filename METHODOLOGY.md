# Methodology: Residual Stream Dynamics in Deep Transformers

## 1. Introduction

The standard Transformer architecture relies on additive residual connections to facilitate gradient flow and enable training of deep networks. However, as models scale to extreme depths, these connections suffer from "information dilution," where the signals from early layers are progressively obscured by the cumulative noise of subsequent transformations. This document is a comparative study of **residual stream mechanisms** for the depth dimension—examining how different architectural choices (simple addition, attention over depth, recurrent gating, and hybrid sliding-window covariance) affect information persistence, gradient flow, and language-model quality on a common Pre‑LN backbone.

---

## 2. The Problem: Dilution and Complexity

### 2.1 Standard Residuals (The Dilution Problem)

In a standard Pre‑LN Transformer, each layer performs the operation:

$$
h_{l+1} = h_l + \text{Sublayer}(\text{LayerNorm}(h_l))
$$

While effective for gradient propagation, this fixed‑weight addition ($1.0 \cdot h_l$) results in a uniform accumulation of all prior outputs. In very deep models, the specific features learned in early layers are "diluted" by the sum of dozens or hundreds of later layers. The model lacks a mechanism to selectively "carry forward" critical information or "filter out" irrelevant updates.

### 2.2 Attention Residuals (The Complexity Problem)

**Attention Residuals (AttnRes)** address dilution by treating the depth dimension as a sequence, using softmax attention to aggregate previous states:

$$
h_{l+1} = \sum_{i=0}^{l} \text{softmax}(q_{l} \cdot k_{i}) v_{i}
$$

While this allows "Random Access" to any previous layer, it introduces a memory and compute bottleneck that scales linearly with depth ($O(L)$). For a 1,000‑layer model, every layer must attend to 1,000 previous representations, making it prohibitively expensive for large‑scale deployment. Furthermore, the depth‑wise softmax creates attention‑sink dynamics that amplify outliers and harm quantisation.

---

## 3. Recurrent Residuals (RR): Gated Working Memory

Recurrent Residuals replace the "Random Access" of attention with a **"Working Memory"** approach. Instead of looking back at every layer, the model maintains a persistent per‑token memory state $\mathbf{m}$ that is recurrently updated as information flows deeper into the network.

### 3.1 Core Philosophy

The RR mechanism treats the entire depth of the Transformer as a single recurrent process. Instead of independent residuals per layer, the model maintains a **persistent memory state** $\mathbf{m}$ that flows through the entire stack alongside the hidden state $\mathbf{h}$.

As the hidden state progresses from Layer 1 to Layer $L$, it continuously interacts with this memory through two gated operations: a **read gate** that injects historical context into the current computation, and an **update gate** that selectively writes new information into the memory. This architecture effectively turns the Transformer into a "Depth‑RNN," where each sublayer is a step in a recurrent transition. Information captured in the very first attention head can be preserved and retrieved in the final feed‑forward network without duplication or dilution.

---

## 4. Mathematical Framework

The RR mechanism consists of two primary operations per sublayer: **Reading** (injection of memory) and **Updating** (revision of memory). All gates use diagonal weights (element‑wise products) for both computational efficiency and inductive alignment with optimal depth‑wise gating.

$$
\boxed{
\left\{
\begin{aligned}
\mathbf{y}_l &= \mathcal{F}_l\!\bigl(\mathrm{LayerNorm}(\mathbf{h}_{l-1})\bigr) \\[4pt]
\mathbf{r}_l &= \sigma\!\bigl(\mathbf{w}_r \odot \mathrm{LayerNorm}(\mathbf{h}_{l-1}) + \mathbf{b}_r\bigr) \qquad \text{(read gate)} \\[4pt]
\mathbf{h}_l &= \mathbf{h}_{l-1} + \mathbf{y}_l + \mathbf{r}_l \odot \bigl(\mathbf{g}_m \odot \mathrm{RMSNorm}(\mathbf{m}_{l-1})\bigr) \\[4pt]
\mathbf{f}_l &= \sigma\!\bigl(\mathbf{w}_f \odot \mathbf{y}_l + \mathbf{b}_f\bigr) \qquad \text{(forget gate)} \\[4pt]
\mathbf{u}_l &= \sigma\!\bigl(\mathbf{w}_u \odot \mathbf{y}_l + \mathbf{b}_u + \mathbf{p}_l\bigr) \qquad \text{(update gate)} \\[4pt]
\mathbf{m}_l &= \mathbf{f}_l \odot \mathbf{m}_{l-1} + \mathbf{u}_l \odot \mathbf{y}_l
\end{aligned}
\right.
}
$$

**Components:**

- $\mathbf{h}_{l-1}$ : hidden state entering layer $l$; $\mathbf{h}_0$ is the token embedding.
- $\mathbf{m}_{l-1}$ : memory state entering layer $l$; $\mathbf{m}_0$ is a learnable initial memory vector (initialized to $\mathbf{0}$).
- $\mathbf{y}_l$ : output of the sublayer (attention or feed‑forward) after layer normalisation.
- $\mathbf{r}_l$ : **read gate**—decides how much memory to inject, per dimension. Computed from the normalised hidden state to prevent depth‑dependent saturation.
- $\mathbf{g}_m$ : **memory gain**—learnable per‑dimension gain applied to the RMS‑normalised memory before injection.
- $\mathbf{f}_l$ : **forget gate**—controls the retention of the old memory state $\mathbf{m}_{l-1}$ per dimension.
- $\mathbf{u}_l$ : **update gate**—controls the accumulation of the new output $\mathbf{y}_l$ per dimension.
- $\mathbf{p}_l$ : **layer position bias**—a learned per‑layer depth bias that gives each layer a position‑specific prior for how aggressively to write into memory.

All weight vectors ($\mathbf{w}_r, \mathbf{w}_f, \mathbf{w}_u, \mathbf{g}_m$) are diagonal, i.e. applied as $\mathbf{w} \odot \mathbf{x}$. This is theoretically motivated: AttnRes’s own ablation shows that per‑channel mixing in depth‑wise attention hurts performance, making diagonal weights not just cheaper but empirically optimal.

### 4.1 Memory Injection (The Read Gate)

At each sublayer, the model decides how much of the persistent memory context should be merged into the current hidden state. This is controlled by the read gate $\mathbf{r}_l$:

$$
\mathbf{r}_l = \sigma\!\bigl(\mathbf{w}_r \odot \mathrm{LayerNorm}(\mathbf{h}_{l-1}) + \mathbf{b}_r\bigr)
$$

$$
\mathbf{h}_l = \mathbf{h}_{l-1} + \mathbf{y}_l + \mathbf{r}_l \odot \bigl(\mathbf{g}_m \odot \mathrm{RMSNorm}(\mathbf{m}_{l-1})\bigr)
$$

The read gate operates on $\mathrm{LayerNorm}(\mathbf{h}_{l-1})$ to remain responsive at all depths. The memory is RMS‑normalised before projection by $\mathbf{g}_m$, so the gain vector controls only directional emphasis, not arbitrary scaling. The memory is read **before** being updated (i.e., using $\mathbf{m}_{l-1}$), ensuring $\mathbf{y}_l$ contributes to $\mathbf{h}_l$ only once through the direct residual path.

### 4.2 Memory Update (Gated Gating)

After the sublayer transformation is computed, the persistent memory is updated using independent forget and update gates:

$$
\mathbf{f}_l = \sigma\!\bigl(\mathbf{w}_f \odot \mathbf{y}_l + \mathbf{b}_f\bigr)
$$

$$
\mathbf{u}_l = \sigma\!\bigl(\mathbf{w}_u \odot \mathbf{y}_l + \mathbf{b}_u + \mathbf{p}_l\bigr)
$$

$$
\mathbf{m}_l = \mathbf{f}_l \odot \mathbf{m}_{l-1} + \mathbf{u}_l \odot \mathbf{y}_l
$$

The forget gate regulates the retention of historical memory, while the update gate regulates the accumulation of new information. By decoupling the gating mechanisms, the model eliminates the zero-sum trade-off inherent in coupled updates (where writing new content forces forgetting old content). This allows early representations to persist across depth while new representations are integrated independently.

---

## 5. Initialisation and Stability

### 5.1 Zero‑Start Protocol

The mechanism is initialised so that at $t=0$ the model behaves **exactly** as a standard Pre‑LN Transformer:

- $\mathbf{g}_m = \mathbf{0}$ — injection term is zero regardless of memory content.
- $\mathbf{b}_r = -3$ — read gate $\mathbf{r}_l \approx 0.047$ (closed, but with non‑zero gradient).
- $\mathbf{b}_f = 3$ — forget gate $\mathbf{f}_l \approx 0.952$ (highly retentive to preserve early layer outputs).
- $\mathbf{b}_u = -2$ — update gate $\mathbf{u}_l \approx 0.119$ (conservative writing).
- $\mathbf{p}_l = \mathbf{0}$ for all $l$ — all layers start with equal write bias.
- $\mathbf{w}_r, \mathbf{w}_f, \mathbf{w}_u \sim \mathcal{N}(0, 0.01^2)$ — minor random asymmetry, negligible next to bias magnitude.
- $\mathbf{m}_0 = \mathbf{0}$ (learnable initial memory, starting from zero).

The standard skip connection $\mathbf{h}_{l-1}$ remains untouched, providing an unchanging gradient highway. As training proceeds, $\mathbf{g}_m$ departs from zero and the gates specialise, letting the memory path activate gradually.

### 5.2 Bounded Norm Dynamics

The memory $\mathbf{m}_l$ is updated with sigmoid-bounded gates $\mathbf{f}_l, \mathbf{u}_l \in [0, 1]^d$. The read path applies RMSNorm to $\mathbf{m}_{l-1}$ before injection, preventing scale explosion and ensuring stability even under additive updates. The hidden state $\mathbf{h}_l$ receives the injection as a strictly additive, gated contribution; the core residual branch remains intact, so gradient flow and activation magnitude stay well‑behaved.

---

| Component | Parameters |
|-----------|------------|
| Read gate weight $\mathbf{w}_r$ | $d$ |
| Read gate bias $\mathbf{b}_r$ | $d$ |
| Memory gain $\mathbf{g}_m$ | $d$ |
| Forget gate weight $\mathbf{w}_f$ | $d$ |
| Forget gate bias $\mathbf{b}_f$ | $d$ |
| Update gate weight $\mathbf{w}_u$ | $d$ |
| Update gate bias $\mathbf{b}_u$ | $d$ |
| Sublayer position biases $\{\mathbf{p}_s\}_{s=1}^S$ | $S \cdot d$ |
| Initial memory $\mathbf{m}_0$ | $d$ |
| **Total** | $(S + 8)d$ |

Here $S$ is the number of residual transitions. For a decoder layer with attention and feed‑forward sublayers, $S = 2L$. A 12‑layer model with $d = 4096$ therefore adds approximately **131k parameters**, completely independent of the main model size. The per‑sublayer compute overhead is $O(d)$ (element‑wise operations), with no attention, no block partitions, and no cached layer outputs.

---

## 6.5 VEGA: Vertical EMA Gated Attention

VEGA combines multi-scale depth memory to capture context at different timescales:

**Multi-Scale Depth Memory:** Instead of a FIFO window, VEGA partitions heads into "fast" and "slow" decay groups within a single linear-attention EMA state. This allows the model to dynamically learn to retain information at different timescales (local vs. global depth context) without the memory overhead of a FIFO buffer.

**EMA State Update:** The state $\mathbf{S} \in \mathbb{R}^{H \times r \times r}$ is updated as an exponential moving average:

$$
\mathbf{S}_l = \boldsymbol{\alpha}_l \odot \mathbf{S}_{l-1} + \boldsymbol{\phi}(\mathbf{k}_l)^{\!\top} \mathbf{v}_l
$$

$$
c_{\text{deep}} = \frac{\boldsymbol{\phi}(\mathbf{q}_l)^\top \mathbf{S}_{l-1}}{\boldsymbol{\phi}(\mathbf{q}_l)^\top \mathbf{z}_{l-1} + \varepsilon}
$$

where $\boldsymbol{\phi}(x) = \text{ELU}(x) + 1$ ensures unconditional stability, and $\boldsymbol{\alpha}_l$ are per-head, per-rank decay gates initialized with logarithmic timescale spacing (from 1-step to $2L$-step memory half-lives).

**Key initialisation choices:**
- Decay gates initialized log-linearly so each head specializes in a different depth horizon.
- Learned Depth Pseudo-Queries (`query_bias`) initialized to zero (Zero-Start compliant).
- Projection matrices $\mathbf{W}_K, \mathbf{W}_Q, \mathbf{W}_V$ initialized orthogonally.
- Memory gain $\mathbf{g}_m = \mathbf{1}$, controlled by a closed read gate at init, so the model starts as a standard Pre-LN Transformer.

---

## 7. Comparison of Approaches

| Feature | Standard Residuals | Attention Residuals | Recurrent Residuals | VEGA |
| :--- | :--- | :--- | :--- | :--- |
| **Logic** | Simple Addition | Depth‑wise Softmax | Gated Recurrency (Decoupled) | FIFO Window + Low-Rank Linear Attn |
| **Information Flow** | Passive Accumulation | Selective Retrieval | Selective Persistence | Local Exact + Deep Covariance |
| **Complexity (per layer)** | $O(d)$ | $O(Ld)$ (full) / $O(Bd)$ (block) | $O(d)$ | $O(Wd + rd)$ |
| **Memory Cost (per token)** | $O(d)$ | $O(Ld)$ (full) / $O(Bd)$ (block) | $O(2d)$ | $O(Wd + nrd_v)$ |
| **Softmax over Depth** | No | Yes | No | No (local only) |
| **Hyperparameters Added** | 0 | Block size, sink tokens, rescaling | 0 | Window size $W$, rank $r$, heads $H$ |
| **Stability** | Norm grows with depth | Attention‑sink outliers | Provably bounded norms | Linear-attn positivity constraint |
| **Initialisation Sensitivity** | Low | Medium (pseudo‑query zero‑init) | None (starts as standard Transformer) | None (Zero-Start compliant) |

---

## 8. Evaluation Metrics: Probing Representation Dynamics

To systematically evaluate whether a residual connection prevents information loss across depth, we define two complementary closed-form probing metrics. Rather than training auxiliary neural probes (which introduces optimization noise and hyperparameters), both metrics are computed in a single pass using Ridge Regression in closed form.

### 8.1 Depth Preservation Score (DPS) & Dilution Resistance Index (DRI)

The **Depth Preservation Score (DPS)** measures the degree to which the *entirety* of an intermediate representation is linearly accessible from the final hidden state.

Let $\mathbf{a}_k^{(i)} \in \mathbb{R}^d$ be the parameter-free normalized activation of layer $k$ for token $i$, and let $\mathbf{s}^{(i)} \in \mathbb{R}^d$ be the final hidden state before the LM head. We fit a linear map $(\mathbf{W}, \mathbf{b})$ to reconstruct $\mathbf{a}_k$ from $\mathbf{s}$:

$$
\mathrm{DPS}(k) = 1 - \frac{\sum_{i=1}^N \|\mathbf{a}_k^{(i)} - (\mathbf{W}\mathbf{s}^{(i)} + \mathbf{b})\|^2}{\sum_{i=1}^N \|\mathbf{a}_k^{(i)} - \bar{\mathbf{a}}_k\|^2}
$$

The projection parameters $\mathbf{W} \in \mathbb{R}^{d \times d}$ and $\mathbf{b} \in \mathbb{R}^d$ are solved in closed form using Ridge Regression with regularizer $\lambda = 1.0$. 

The **Dilution Resistance Index (DRI)** is the average DPS over the first half of the network, summarizing how well early information is preserved:

$$
\mathrm{DRI} = \frac{1}{\lfloor L/2 \rfloor} \sum_{k=1}^{\lfloor L/2 \rfloor} \mathrm{DPS}(k)
$$

> [!NOTE]
> **Linearity Bound**: DPS is a *lower bound* on information preservation. A low DPS means the representation is not linearly accessible, though it may still be preserved non-linearly.

---

### 8.2 Gradient Preservation Score (GPS) & Gradient Preservation Index (GPI)

DPS measures whether *all* early layer information is preserved. However, a healthy deep network should abstract away low-level features. The **Gradient Preservation Score (GPS)** measures whether the *task-relevant* subspace of the early layer is preserved.

If we apply the LM head directly to the early representation $\mathbf{a}_k$, we obtain early logits and an early loss. The gradient of this early loss points in the direction that would most improve the prediction at layer $k$. If this corrective task direction is recoverable from the final state $\mathbf{s}$, the final state still carries the task-relevant information.

For each token $i$, the implicit early gradient $\mathbf{g}_k^{(i)}$ is defined as:

$$
\mathbf{g}_k^{(i)} = \mathbf{W}_{\text{head}}^\top (\text{softmax}(\mathbf{a}_k^{(i)} \mathbf{W}_{\text{head}}) - \mathbf{y}^{(i)}) \in \mathbb{R}^d
$$

where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{d \times V}$ are the LM head weights, and $\mathbf{y}^{(i)}$ is the one-hot target token label.

Using closed-form Ridge Regression ($\lambda = 1.0$), we solve for the mapping $(\mathbf{W}_g, \mathbf{b}_g)$ to reconstruct $\mathbf{g}_k$ from $\mathbf{s}$:

$$
\mathrm{GPS}(k) = 1 - \frac{\sum_{i=1}^N \|\mathbf{g}_k^{(i)} - (\mathbf{W}_g \mathbf{s}^{(i)} + \mathbf{b}_g)\|^2}{\sum_{i=1}^N \|\mathbf{g}_k^{(i)} - \bar{\mathbf{g}}_k\|^2}
$$

The **Gradient Preservation Index (GPI)** is the average GPS over the first half of the network:

$$
\mathrm{GPI} = \frac{1}{\lfloor L/2 \rfloor} \sum_{k=1}^{\lfloor L/2 \rfloor} \mathrm{GPS}(k)
$$

---

### 8.3 The Golden Pairing Rule

Neither metric is sufficient on its own. We evaluate architectures using the triple index **(DRI, GPI, Perplexity)**:

| DRI | GPI | Perplexity vs. Baseline | Interpretation |
| :--- | :--- | :--- | :--- |
| High | High | Better or Equal | **Successful preservation**: Task-relevant information is preserved linearly, leading to equal or improved performance. |
| High | Low | Better or Equal | **Healthy abstraction**: The model discards task-irrelevant raw details in later layers in favor of higher-level semantic features. |
| High | Low | Worse | **Cluttering**: The residual mechanism forces the model to retain raw features that clutter the representation and harm task performance. |
| Low | Low | Worse | **Classic dilution**: Crucial early-layer information is diluted and lost, leading to degraded performance. |
| Low | High | - | **Targeted preservation**: The raw representation is heavily transformed/compressed, but the specific task-relevant gradient direction remains linearly readable. |

---

## 9. Conclusion

Each residual mechanism studied here makes a distinct trade-off between expressivity, memory cost, and stability. Standard residuals are a free baseline. Attention Residuals offer selective random-access depth retrieval at $O(Ld)$ cost. Recurrent Residuals provide $O(d)$ persistent working memory with provably bounded norms. VEGA combines both local exactness and long-range covariance retrieval at $O(Wd + rd)$ cost, bridging the gap between the two extremes. The shared evaluation framework—DRI, GPI, and perplexity—provides a principled lens for comparing these designs on both information-preservation and downstream language-model quality.
