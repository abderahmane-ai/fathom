# Methodology: Recurrent Residual Mechanisms for Deep Transformers

## 1. Introduction

The standard Transformer architecture relies on additive residual connections to facilitate gradient flow and enable training of deep networks. However, as models scale to extreme depths, these connections suffer from "information dilution," where the signals from early layers are progressively obscured by the cumulative noise of subsequent transformations. This document outlines the methodology behind **Recurrent Residuals (RR)**—a novel, gated memory mechanism designed to preserve information persistence across the depth dimension with constant $O(1)$ complexity, operating on a standard Pre‑LN backbone without additional stabilisation tricks.

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

## 3. The Solution: Recurrent Residuals (RR)

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
\mathbf{u}_l &= \sigma\!\bigl(\mathbf{w}_u \odot \mathbf{y}_l + \mathbf{b}_u + \mathbf{p}_l\bigr) \qquad \text{(update gate)} \\[4pt]
\mathbf{m}_l &= \mathbf{u}_l \odot \mathbf{y}_l + (\mathbf{1} - \mathbf{u}_l) \odot \mathbf{m}_{l-1}
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
- $\mathbf{u}_l$ : **update gate**—controls the blend between the new output $\mathbf{y}_l$ and the old memory $\mathbf{m}_{l-1}$ in the EMA.
- $\mathbf{p}_l$ : **layer position bias**—a learned per‑layer depth bias that gives each layer a position‑specific prior for how aggressively to write into memory.

All weight vectors ($\mathbf{w}_r, \mathbf{w}_u, \mathbf{g}_m$) are diagonal, i.e. applied as $\mathbf{w} \odot \mathbf{x}$. This is theoretically motivated: AttnRes’s own ablation shows that per‑channel mixing in depth‑wise attention hurts performance, making diagonal weights not just cheaper but empirically optimal.

### 4.1 Memory Injection (The Read Gate)

At each sublayer, the model decides how much of the persistent memory context should be merged into the current hidden state. This is controlled by the read gate $\mathbf{r}_l$:

$$
\mathbf{r}_l = \sigma\!\bigl(\mathbf{w}_r \odot \mathrm{LayerNorm}(\mathbf{h}_{l-1}) + \mathbf{b}_r\bigr)
$$

$$
\mathbf{h}_l = \mathbf{h}_{l-1} + \mathbf{y}_l + \mathbf{r}_l \odot \bigl(\mathbf{g}_m \odot \mathrm{RMSNorm}(\mathbf{m}_{l-1})\bigr)
$$

The read gate operates on $\mathrm{LayerNorm}(\mathbf{h}_{l-1})$ to remain responsive at all depths. The memory is RMS‑normalised before projection by $\mathbf{g}_m$, so the gain vector controls only directional emphasis, not arbitrary scaling. The memory is read **before** being updated (i.e., using $\mathbf{m}_{l-1}$), ensuring $\mathbf{y}_l$ contributes to $\mathbf{h}_l$ only once through the direct residual path.

### 4.2 Memory Update (The Update Gate)

After the sublayer transformation is computed, the persistent memory is updated using an Exponential Moving Average (EMA) logic, controlled by the update gate $\mathbf{u}_l$:

$$
\mathbf{u}_l = \sigma\!\bigl(\mathbf{w}_u \odot \mathbf{y}_l + \mathbf{b}_u + \mathbf{p}_l\bigr)
$$

$$
\mathbf{m}_l = \mathbf{u}_l \odot \mathbf{y}_l + (\mathbf{1} - \mathbf{u}_l) \odot \mathbf{m}_{l-1}
$$

The update gate depends on the current sublayer output (content) and a learned per‑sublayer depth bias $\mathbf{p}_l$. Because the blend is a convex combination ($\mathbf{u}_l + (\mathbf{1} - \mathbf{u}_l) = \mathbf{1}$), each dimension of $\mathbf{m}_l$ remains bounded by the extreme values of past $\mathbf{y}$ vectors. No softmax, no competition across layers or channels—therefore no attention‑sink dynamics and no outlier amplification.

---

## 5. Initialisation and Stability

### 5.1 Zero‑Start Protocol

The mechanism is initialised so that at $t=0$ the model behaves **exactly** as a standard Pre‑LN Transformer:

- $\mathbf{g}_m = \mathbf{0}$ — injection term is zero regardless of memory content.
- $\mathbf{b}_r = -3$ — read gate $\mathbf{r}_l \approx 0.047$ (closed, but with non‑zero gradient).
- $\mathbf{b}_u = -2$ — update gate $\mathbf{u}_l \approx 0.119$ (conservative writing, memory half‑life ≈ 5.5 layers).
- $\mathbf{p}_l = \mathbf{0}$ for all $l$ — all layers start with equal write bias.
- $\mathbf{w}_r, \mathbf{w}_u \sim \mathcal{N}(0, 0.01^2)$ — minor random asymmetry, negligible next to bias magnitude.
- $\mathbf{m}_0 = \mathbf{0}$ (learnable initial memory, starting from zero).

The standard skip connection $\mathbf{h}_{l-1}$ remains untouched, providing an unchanging gradient highway. As training proceeds, $\mathbf{g}_m$ departs from zero and the gates specialise, letting the memory path activate gradually.

### 5.2 Bounded Norm Dynamics

The memory $\mathbf{m}_l$ is a convex combination of past $\mathbf{y}_i$, each normalised by LayerNorm. Hence $\|\mathbf{m}_l\|$ cannot diverge. The injection path further applies RMSNorm to $\mathbf{m}_{l-1}$, preventing $\mathbf{g}_m$ from amplifying scale. The hidden state $\mathbf{h}_l$ receives the injection as a strictly additive, gated contribution; the core residual branch remains intact, so gradient flow and activation magnitude stay well‑behaved without any external rescaling or DeepNorm constants.

---

## 6. Complexity and Parameter Count

| Component | Parameters |
|-----------|------------|
| Read gate weight $\mathbf{w}_r$ | $d$ |
| Read gate bias $\mathbf{b}_r$ | $d$ |
| Memory gain $\mathbf{g}_m$ | $d$ |
| Update gate weight $\mathbf{w}_u$ | $d$ |
| Update gate bias $\mathbf{b}_u$ | $d$ |
| Sublayer position biases $\{\mathbf{p}_s\}_{s=1}^S$ | $S \cdot d$ |
| Initial memory $\mathbf{m}_0$ | $d$ |
| **Total** | $(S + 6)d$ |

Here $S$ is the number of residual transitions. For a decoder layer with attention and feed‑forward sublayers, $S = 2L$. A 12‑layer model with $d = 4096$ therefore adds approximately **123k parameters**, completely independent of the main model size. The per‑sublayer compute overhead is $O(d)$ (element‑wise operations), with no attention, no block partitions, and no cached layer outputs.

---

## 7. Comparison of Approaches

| Feature | Standard Residuals | Attention Residuals | Recurrent Residuals |
| :--- | :--- | :--- | :--- |
| **Logic** | Simple Addition | Depth‑wise Softmax | Gated Recurrency (EMA) |
| **Information Flow** | Passive Accumulation | Selective Retrieval | Selective Persistence |
| **Complexity (per layer)** | $O(d)$ | $O(Ld)$ (full) / $O(Bd)$ (block) | $O(d)$ |
| **Memory Cost (per token)** | $O(d)$ | $O(Ld)$ (full) / $O(Bd)$ (block) | $O(2d)$ |
| **Softmax over Depth** | No | Yes | No |
| **Hyperparameters Added** | 0 | Block size, sink tokens, rescaling | 0 |
| **Stability** | Norm grows with depth | Attention‑sink outliers | Provably bounded norms |
| **Initialisation Sensitivity** | Low | Medium (pseudo‑query zero‑init) | None (starts as standard Transformer) |

---

## 8. Conclusion

Recurrent Residuals provide a scalable, $O(d)$ solution to the information dilution problem in deep Transformers. By integrating gated working memory with a principled initialisation protocol, the architecture enables the persistence of early‑layer signals through hundreds of layers without the quadratic memory costs and stability risks associated with full attention‑over‑depth mechanisms. The design introduces no new hyperparameters, requires no special normalisation schemes beyond standard Pre‑LN, and can be dropped into any Transformer as a direct replacement for the vanilla residual connection.
