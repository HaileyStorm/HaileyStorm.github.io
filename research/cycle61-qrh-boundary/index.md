---
layout: page
title: A zero-free half-plane with boundary 874861/1000000
permalink: /research/cycle61-qrh-boundary/
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
    <span class="paper-download__detail">PDF · 24 pages</span>
  </a>
  <a class="paper-download" href="paper.tex">TeX source</a>
  <a class="paper-download" href="paper.md">Markdown source</a>
</nav>

This continues the research behind the [single-compensating-prime proof](/research/single-compensating-prime/) and [quartic boundary](/research/quartic-qrh-boundary/). It gives **c = 0.125139**, or the exact zero-free half-plane **Re(s) > 0.874861**.

The theorem covers every complex Dirichlet character of nonzero modulus, the Riemann zeta function, and the stated finite-order Hecke family over the Eisenstein field. The principal poles are excluded, and the boundary is strict.

The argument combines earlier moment estimates and source machinery with Gao's local refinement, extends the required density range, and supplies a common probe for the final zero-exclusion argument. The paper follows the completed Lean construction, including its source normalization, choice order and full error estimate.

## Formalization and verification

The unconditional endpoint and the Dirichlet, zeta and Hecke declarations compile in Lean 4.34.1 with only the standard axioms. The complete clean rebuild passed for all 3,695 source modules, with no submitted objects reused. The original Dirichlet, zeta and Hecke proof closures passed Comparator and all three kernels: Lean, NanoDa and con-ron. The bundled Candidate also passed a fresh 194-module compilation, the canonical Dirichlet Comparator and all three kernels.

Both local replays used a documented native judge adapter because this host refused the standard nested namespace setup. They retained the trusted Comparator, protected inputs and official per-kernel sandboxes. The [verification record](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/fb6d4fd11aeaff0b37dd96bad3b274ce28a5c08b/proofs/cycle61-c0125139/Verification.md) describes those checks and their limits. The separate protected check of the submitted commit is running in [PR #42](https://github.com/GTimG/QuasiRiemannTracker/pull/42); consult the PR for its current status.

## Authorship and sources

I initiated and directed the research, chose the targets and sources, and organized the assembly and checking of the argument. The detailed mathematical derivations, substantial drafting and formalization were AI-assisted. I am the sole human author; this does not attribute every derivation produced within the project to me personally.

The proof credits OpenAI's original seven-eighths framework, Akash Levy's weighted-numerator work, Gao's local refinement and the earlier AI-assisted work in this research program. [Read the detailed contribution and attribution account](contributions/), including my role in directing the research and bringing its sources together. The paper and new code are available under Apache-2.0, with upstream attribution and licenses retained.
