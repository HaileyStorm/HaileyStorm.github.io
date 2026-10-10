---
layout: page
title: A zero-free half-plane from a single compensating prime
permalink: /research/single-compensating-prime/
---

[Read the paper — PDF, 36 pages](paper.pdf) · [TeX source](paper.tex) · [Markdown source](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/94fbacd5db79f62ef9bd316a6a41ac51627154b5/proofs/single-prime-29-33-20261010/manuscript/zeta_zero_free_4_33_self_contained.md)

The appeal of this manuscript is how compact the self-contained argument is for **c = 4/33**, giving a zero-free half-plane **Re(s) > 29/33**, with the usual pole exclusions. [Shiva Kintali’s proof for c = 1/48](https://shivakintali.github.io/papers/QRH.pdf) is what started me on this. His manuscript is **27 pages**; this one is **36 pages** for c = 4/33.

This is a **single-compensating-prime alternative proof, with an independently usable unconditional moving-radical mixed moment**. It is a non-record result: the 29/33 boundary is weaker than 7/8. The manuscript treats the Riemann zeta function, all Dirichlet characters of positive modulus, and finite-order Hecke characters over the Eisenstein field. Appendix D.3 supplies the unconditional mixed-moment theorem separately from the zero-free endpoint.

## Lean formalization

I started the formalization to try out FrankenLean, then moved it to ordinary Lean, with substantial OpenAI Codex assistance and foundations from [OpenAI/math](https://github.com/openai/math/blob/main/lean/docs/003.md).

The formalization has about **24,000 local Lean lines and 703 theorems**, on top of about 478,000 lines of reused OpenAI source. Mathlib and Lean itself are additional dependencies.

The ordinary-Lean audit passed for all 1,087 local declarations. **Lean, NanoDa and con-ron also accepted the exported proof closures for the four main statements**: zeta, Dirichlet, Hecke, and the standalone Appendix D.3 estimate. These are local contributor checks; maintainer verification is still pending in QuasiRiemannTracker.

- [QuasiRiemannTracker submission](https://github.com/GTimG/QuasiRiemannTracker/pull/8)
- [Formalization and reproduction instructions](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/94fbacd5db79f62ef9bd316a6a41ac51627154b5/proofs/single-prime-29-33-20261010/README.md)
- [Verification scope, logs and checker details](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/94fbacd5db79f62ef9bd316a6a41ac51627154b5/proofs/single-prime-29-33-20261010/Verification.md)
- [Attribution and source provenance](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/94fbacd5db79f62ef9bd316a6a41ac51627154b5/proofs/single-prime-29-33-20261010/NOTICE.md)

The supplied PDF and TeX are preserved unchanged. The notices distinguish formalization credit from mathematical authorship and retain the licenses for reused sources.
