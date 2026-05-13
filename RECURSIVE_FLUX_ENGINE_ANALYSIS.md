# Recursive Flux Engine: Technical Breakdown

## 1. Mathematical Flow: Discrete to Continuous Transition

The Recursive Flux Engine bridges the gap between discrete sequence modeling and continuous-time dynamical systems to identify logical inflection points.

### Mamba-3 Discrete State Update
The Mamba-3 core operates on a discretized state-space model:
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t + Dx_t$$
Where $\bar{A}$ and $\bar{B}$ are discretized via the Exponential-Trapezoidal rule, which provides second-order accuracy compared to the standard Euler discretization in earlier SSMs.

### LNN Continuous-Time Derivative
The Liquid Neural Network (LNN) layer receives the hidden state $h_t$ from Mamba. Instead of a fixed transformation, it treats the state evolution as an Ordinary Differential Equation (ODE):
$$\frac{dh}{dt} = -[f(x, t)] \odot h(t) + S(t)$$
Where:
- $f(x, t)$ is the "Liquid" time-constant that adapts to the input data.
- $S(t)$ is the external system stimulus derived from the current Mamba state.

### The Transition
The engine maps the discrete jump $\Delta h = h_t - h_{t-1}$ into the LNN stimulus. A "Semantic Phase Shift" is detected when the LNN's internal state variance $\sigma(h(t))$ exceeds a learned threshold $\tau$, indicating that the model has encountered a token requiring high-complexity logical branching (e.g., the start of a mathematical proof or a shift in intent).

## 2. Efficiency Gains: Linear Scaling vs. KV Cache

### The Transformer Bottleneck (2026 Context)
A 2026-era Transformer with 1 million tokens requires a KV cache that scales linearly with sequence length:
- **Memory**: $\mathcal{O}(N)$ for KV cache storage, leading to Gigabytes of VRAM just for state.
- **Compute**: $\mathcal{O}(N^2)$ (or $\mathcal{O}(N)$ for linear attention, but with high constant overhead and precision loss).

### Recursive Flux Engine Advantage
- **Linear Compute**: SSMs (Mamba-3) and Linear Recurrences (RWKV-7) maintain $\mathcal{O}(N)$ complexity.
- **Constant VRAM**: The hidden state $h_t$ has a fixed size regardless of sequence length.
- **Adaptive Depth**: Unlike Transformers which have a fixed depth $L$, the MoR wrapper allows "difficult" tokens to be processed $L \times R$ times (where $R$ is the recursion count) while "easy" tokens pass through once, maximizing FLOP efficiency.

## 3. The "Frontier Killer" Logic: 20B vs. 500B

The Recursive Flux Engine achieves 500B-level reasoning with 20B parameters through two key mechanisms:

### Mixture-of-Recursions (MoR)
Standard LLMs are "shallow" in their thinking—they apply the same number of layers to the word "the" as they do to a complex logical deduction. MoR enables **Adaptive Compute Time (ACT)**. The model reuses its shared weights recursively. A 20B model with 10 recursive passes effectively behaves like a 200B+ model for that specific token, concentrating its intelligence where it matters most.

### RWKV-7 HeadQK Mechanism
Associative recall is the Achilles' heel of linear models. The HeadQK mechanism in RWKV-7 introduces a Query-Key gating system within the linear recurrence:
$$S_t = diag(W_k \cdot x_t) \cdot S_{t-1} + (W_q \cdot x_t) \otimes (W_v \cdot x_t)$$
By using the **Generalized Delta Rule**, HeadQK allows the model to "write" specific facts into its hidden state with high precision and "read" them back later, matching the 99% associative recall accuracy of Transformers.

## 4. Stability Strategy

Managing gradients across Discrete (Mamba) $\rightarrow$ Continuous (LNN) $\rightarrow$ Recursive (MoR) interfaces requires a three-pronged approach:

1.  **State Normalization**: Apply GroupNorm or RMSNorm to the Mamba hidden state before it enters the LNN to prevent exponential growth during ODE integration.
2.  **Bypass Gating**: Implement a residual connection around the LNN and MoR blocks. If the LNN fails to solve the ODE within the time step, the "discrete" path remains viable.
3.  **Recursive Gradient Clipping**: Use "Stochastic Depth" during training in the MoR core—randomly varying the number of recursive passes—to ensure the model learns to maintain a stable state regardless of recursion depth.
