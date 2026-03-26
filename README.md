# Prototype Summary

**Decoding Market Spillover via Multi-Relational Graph Transformers and RoPE**
Weihang Huang (wh2607), Astrid Teo (st3750) | March 2026

---

## What We Built

An LSTM + Graph Transformer pipeline that uses **WIRE (graph-structural rotary encodings)** to inject stock relationship graphs into self-attention. Tested on 100 S&P 500 stocks, 2015–2025, with 9 features (OHLCV + fundamentals).

---

## Key Findings

**1. WIRE works mechanically, but the signal is too weak to measure at this scale.**
- Ablation (3 configs × 5 seeds): LSTM IC = −0.000 ± 0.015, Sector WIRE IC = −0.011 ± 0.018, Sector+Supply WIRE IC = −0.004 ± 0.014
- All pairwise t-tests: p > 0.49. No significant differences.
- With N=100 and 9 features, IC lives entirely within the ±0.03 noise band.

**2. ω initialization and learning rate are critical.**
- Default init (0.1) → rotation angles ≈ 0 → uniform attention (ratio = 1.00×). Model ignores the graph entirely.
- Fix: init = 1.0, LR × 10, no weight decay on ω. Attention ratio drops to ~0.93–0.97×, confirming sector differentiation.

**3. Graph density has a hard threshold.**
- Sector graph (541 edges, density 0.11): clean spectral clusters, useful.
- Supply chain (310 edges, density 0.06): usable.
- TNIC competitors (68 edges, density 0.014): too sparse → noisy eigenvectors → degrades performance.
- Ownership/13F: similarly sparse. Excluded.
- **Rule of thumb: density < 0.02 → do not use.**

**4. Z-score normalization destroys the signal.**
- Cross-sectional z-scoring makes target variance = 1.0, but the predictable component is tiny.
- Model learns constant-zero output. Raw returns preserve learnable trends.

**5. Scaling to 500+ stocks is non-trivial.**
- Small-cap stocks bring missing data and extreme returns that destabilize attention.
- Need quality filters (min trading days, liquidity) before scaling.

---

## What We'll Do Next

| Priority | Task | Why |
|----------|------|-----|
| **High** | Expand to 300–500 stocks with quality filtering | Reduce IC noise, enable statistical significance |
| **High** | Add 50+ alpha features (technical indicators, momentum, earnings) | More signal for LSTM to extract |
| **High** | GPU training | Current CPU bottleneck limits model/data scale |
| **Medium** | DGT baseline (Pearson correlation graph) | Data-driven graph comparison |
| **Medium** | Time-varying graphs (quarterly TNIC, rolling correlation) | Static graphs miss evolving relationships |
| **Medium** | Regime analysis (COVID, 2022 rate hikes) | Test if graph attention shifts during stress |
| **Lower** | Fama-French alpha, long-short portfolio backtest | Portfolio-level validation |
| **Lower** | Hyperparameter sweep (Optuna) | Systematic optimization of spectral dims, heads, ω LR |

---

## Process Discoveries

- Single-run IC comparisons on 100 stocks are meaningless. Multi-seed + paired t-tests are the minimum bar.
- Adaptive spectral allocation (by eigenvalue energy) is a clean way to distribute capacity across graphs of different quality.
- Gated residual connections are essential — they let the model gracefully ignore a weak graph signal instead of being forced to use it.
- Pre-filtering large datasets (648MB → 5MB, 2.5GB → 184MB) was necessary for Colab feasibility.
- The architecture is sound. The bottleneck is data scale and feature richness, not model design.
