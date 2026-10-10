---
layout: page
title: A zero-free half-plane from a single compensating prime
permalink: /research/single-compensating-prime/
---

**Hailey Collet** · [HaileyCollet@gmail.com](mailto:HaileyCollet@gmail.com)

<style>
.paper-downloads {
  display: grid;
  grid-template-columns: 1.35fr 1fr 1.2fr;
  gap: 12px;
  margin: 24px 0 32px;
}
.paper-downloads .paper-download {
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80px;
  padding: 14px 18px;
  border: 2px solid #c5d5ee;
  border-radius: 8px;
  background: #f3f7ff;
  color: #174b99;
  font-size: 18px;
  font-weight: 600;
  line-height: 1.3;
  text-align: center;
  text-decoration: none;
}
.paper-downloads .paper-download:hover {
  background: #e5eeff;
  border-color: #174b99;
}
.paper-downloads .paper-download:focus-visible {
  outline: 3px solid #122c50;
  outline-offset: 4px;
}
.paper-downloads .paper-download--primary {
  border-color: #1d4ed8;
  background: #1d4ed8;
  color: #fff;
  font-size: 22px;
  font-weight: 700;
}
.paper-downloads .paper-download--primary:hover {
  border-color: #1e40af;
  background: #1e40af;
}
.paper-downloads .paper-download__detail {
  margin-top: 4px;
  font-size: 14px;
  font-weight: 400;
}
@media (max-width: 540px) {
  .paper-downloads { grid-template-columns: 1fr 1fr; }
  .paper-downloads .paper-download { min-height: 64px; padding: 14px 10px; font-size: 16px; }
  .paper-downloads .paper-download--primary { grid-column: 1 / -1; min-height: 84px; font-size: 22px; }
}
</style>
<nav class="paper-downloads" aria-label="Paper downloads">
  <a class="paper-download paper-download--primary" href="paper.pdf">
    <span>Read the paper</span>
    <span class="paper-download__detail">PDF · 36 pages</span>
  </a>
  <a class="paper-download" href="paper.tex">TeX source</a>
  <a class="paper-download" href="https://github.com/HaileyStorm/QuasiRiemannTracker/blob/42e0e0e9b8b04e75bfa8132806a11f2250411e74/proofs/single-prime-29-33-20261010/manuscript/zeta_zero_free_4_33_self_contained.md">Markdown source</a>
</nav>

The appeal of this manuscript is how compact the self-contained argument is for **c = 4/33**, giving a zero-free half-plane **Re(s) > 29/33**, with the usual pole exclusions. [Shiva Kintali’s proof for c = 1/48](https://shivakintali.github.io/papers/QRH.pdf) is what started me on this. His manuscript is **27 pages**; this one is **36 pages** for c = 4/33.

This is a **single-compensating-prime alternative proof, with an independently usable unconditional moving-radical mixed moment**. It is a non-record result: the 29/33 boundary is weaker than 7/8. The manuscript treats the Riemann zeta function, all Dirichlet characters of positive modulus, and finite-order Hecke characters over the Eisenstein field. Appendix D.3 supplies the unconditional mixed-moment theorem separately from the zero-free endpoint.

## Lean formalization

I started the formalization to try out FrankenLean, then moved it to ordinary Lean, with substantial OpenAI Codex assistance and foundations from [OpenAI/math](https://github.com/openai/math/blob/main/lean/docs/003.md).

The formalization has about **24,000 local Lean lines and 703 theorems**, on top of about 478,000 lines of reused OpenAI source. Mathlib and Lean itself are additional dependencies.

The ordinary-Lean audit passed for all 1,087 local declarations. **Lean, NanoDa and con-ron also accepted the exported proof closures for the four main statements**: zeta, Dirichlet, Hecke, and the standalone Appendix D.3 estimate. These are local contributor checks; maintainer verification is still pending in QuasiRiemannTracker.

- [QuasiRiemannTracker submission](https://github.com/GTimG/QuasiRiemannTracker/pull/8)
- [Formalization and reproduction instructions](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/42e0e0e9b8b04e75bfa8132806a11f2250411e74/proofs/single-prime-29-33-20261010/README.md)
- [Verification scope, logs and checker details](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/42e0e0e9b8b04e75bfa8132806a11f2250411e74/proofs/single-prime-29-33-20261010/Verification.md)
- [Attribution and source provenance](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/42e0e0e9b8b04e75bfa8132806a11f2250411e74/proofs/single-prime-29-33-20261010/NOTICE.md)

I developed the manuscript with assistance from OpenAI’s GPT-6 Pro. The notices credit the reused sources and retain their licenses.
