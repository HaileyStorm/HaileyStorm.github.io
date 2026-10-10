---
layout: page
title: The quartic QRH boundary
permalink: /research/quartic-qrh-boundary/
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
    <span class="paper-download__detail">PDF · 58 pages</span>
  </a>
  <a class="paper-download" href="paper.tex">TeX source</a>
  <a class="paper-download" href="https://github.com/HaileyStorm/QuasiRiemannTracker/blob/58344dfdbe756cf2f743da1908ffe0582418cf5b/proofs/cycle25-quartic-20261010/manuscript/paper.md">Markdown source</a>
</nav>

This grew out of my [single-compensating-prime proof for c = 4/33](/research/single-compensating-prime/). The new argument gives **c ≈ 0.1251133529600359**, or a zero-free half-plane **Re(s) > 0.8748866470399641…**. The exact boundary is algebraic: b₀ = 11/12 − ℓ₀/4, where ℓ₀ is the unique root in [1/6, 1/5] of

**927ℓ₀⁴ − 3135ℓ₀³ + 2433ℓ₀² + 275ℓ₀ − 100 = 0.**

The theorem covers the Riemann zeta function, every Dirichlet character of positive modulus, and finite-order Hecke characters of trivial infinite type over the Eisenstein field, with the usual principal pole exceptions. The boundary itself is not included.

For the QuasiRiemannTracker submission, the exact result is rounded outward to **θ = 683505193/781250000 = 0.87488664704**, giving **c = 97744807/781250000**. The rounding loses less than 4 × 10⁻¹⁴.

## Formalization and verification

**Lean, NanoDa and con-ron have all accepted the proof.** Comparator also checked the seven claimed statements and their definitions, allowing only the standard Lean axioms. The QuasiRiemannTracker submission requests acceptance as **framework-verified**; its catalogue status awaits maintainer review.

The argument builds on OpenAI’s original 7/8 theorem, Akash Levy’s weighted-numerator proof, and the unconditional zero-slot moment from the 4/33 formalization. It adds the variable-κ moment estimate, physical numerator bounds and exact quartic optimization needed to reach this boundary. The final theorem has no unproved analytic-estimate premise.

- [QuasiRiemannTracker submission](https://github.com/GTimG/QuasiRiemannTracker/pull/9)
- [Lean source and reproduction instructions](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/58344dfdbe756cf2f743da1908ffe0582418cf5b/proofs/cycle25-quartic-20261010/README.md)
- [Exact statements and verification evidence](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/58344dfdbe756cf2f743da1908ffe0582418cf5b/proofs/cycle25-quartic-20261010/Verification.md)
- [Authorship and source attribution](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/58344dfdbe756cf2f743da1908ffe0582418cf5b/proofs/cycle25-quartic-20261010/PROVENANCE.md)

I directed the research and developed the manuscript with substantial AI assistance, mainly GPT6-Pro and non-Pro Astra, with contributions from GPT 6.1 Sol, Grok 4.7 and Gemini 3.1 Pro. The attribution page explains the human and AI contributions and credits the reused mathematics. The paper and new code are available under Apache-2.0.
