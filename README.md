# GOOSE: Anisotropic Speculation Trees for Training-Free Speculative Decoding

Official repository for the COLM 2026 paper
**"Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding"**
Tao Jin, Phuong Minh Nguyen, Naoya Inoue — Japan Advanced Institute of Science and Technology (JAIST).

## About

Speculative decoding organizes draft tokens as a tree, and a fixed verification budget forces a trade-off
between depth (longer paths) and breadth (more fallbacks). Existing training-free methods draft from a
single token source and shape their trees without distinguishing candidate quality across origins.

GOOSE starts from the observation that two common training-free sources differ sharply in acceptance rate:
*n*-gram matches copied from the context are accepted 2–18× more often (median ≈6×) than statistical
predictions recycled from prior forward passes. When such a gap exists, the optimal tree is **anisotropic** —
reliable tokens form a deep **spine**, unreliable ones spread as wide **branches** at every spine node.
The resulting tree provably accepts at least as many tokens per step as either source alone.

Across five LLMs (7B–33B) and five benchmarks, GOOSE achieves **1.9–4.3× lossless speedup** and outperforms
equal-budget isotropic trees by **12–33%**, with no training and no auxiliary draft model.

## Status

**The code release is in preparation and will be published in this repository before the conference.**
Watch or star this repository to be notified when it lands.

Planned contents:

- the GOOSE decoder (spine-tree construction, tree-attention verification, adjacency table)
- evaluation scripts for the five benchmarks reported in the paper
- configuration files reproducing the main results

## Citation

A BibTeX entry will be added here once the proceedings entry is final.

## Contact

Tao Jin — `morgan@jaist.ac.jp`
