---
layout: page
title: Contributions and attribution
permalink: /research/cycle61-qrh-boundary/contributions/
---

**Hailey Collet** · [HaileyCollet@gmail.com](mailto:HaileyCollet@gmail.com)

[Back to the paper and downloads](/research/cycle61-qrh-boundary/)

I initiated and directed the research program that produced this proof. My contribution includes choosing its targets and sources, proposing questions and approaches, reconciling assumptions, and deciding which arguments to retain, check and combine. I am the sole human author of the resulting manuscript and formalization package. The detailed derivations, substantial drafting and Lean implementation were developed with AI assistance.

## My role in the research

The work required decisions about the intended claim, which earlier results could support it, and how the separate pieces should fit together.

- **Choosing the targets and directing the investigation.** I selected the research targets and sources, proposed questions and approaches, and directed the continuing work on the zero-free boundary. The final result preserves the all-Dirichlet scope, the separate zeta and Hecke comparisons, and the principal-pole exception.
- **Selecting and combining arguments.** I decided what to retain, check and combine as the program developed. Its earlier work supplied reflection, variable-moment, detector and physical-source machinery. Those components form part of the foundation used by the present proof.
- **Bringing in outside work.** I brought Gao's local refinement into the program and asked for its contribution to remain explicit. Integrating it meant reconciling its assumptions and source conventions with the existing argument, while retaining the original contributors' credit.
- **Research and editorial judgment.** I selected or adapted research prompts, including some written with AI assistance, and directed the assembly and checking of the argument. The final manuscript, source package and verification account distinguish the mathematical claim from the evidence available for it.

These contributions describe my role in directing and integrating the research. The mathematical developments credited to this program include detailed derivations produced with AI assistance. In particular, I did not personally derive or write the earlier 4/33 proof.

## What this research program contributed

The earlier [single-compensating-prime work](/research/single-compensating-prime/) and [quartic-boundary work](/research/quartic-qrh-boundary/) developed the reflection, variable moments, physical detector and source architecture used here. The supplied proof packet credits that earlier QRH work to my program. Its attribution carries forward into this result, alongside the underlying OpenAI and Akash Levy contributions.

The later development extends the required height-uniform density range from **[17/20, 7/8] to [33/40, 7/8]**, enlarges the local numerical region, and supplies the rational endpoint **c = 0.125139**. Cycle61 completes the uniform main and error estimates, the common-probe argument and the final zero-exclusion assembly in Lean. The paper explains the source normalization, order of choices and full error estimate used by that construction.

This is an account of the contributed research and its development. The older numerical endpoints are not proofs of the new endpoint; the current result has its own source and verification records.

## AI assistance

Detailed analytic derivations and substantial drafting were developed with AI assistance, mainly GPT6-Pro and non-Pro Astra, with contributions from GPT 6.1 Sol, Grok 4.7 and Gemini 3.1 Pro. Codex also assisted with Lean formalization and integration. These are the tools used across the research program, as recorded in my authorship account; they are not a lemma-by-lemma assignment of credit.

AI review and proof verification serve different purposes. The [verification record](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/fb6d4fd11aeaff0b37dd96bad3b274ce28a5c08b/proofs/cycle61-c0125139/Verification.md) identifies the actual compilation, statement-comparison and kernel checks. The [submission PR](https://github.com/GTimG/QuasiRiemannTracker/pull/42) reports the separate protected tracker check.

## Upstream mathematics and implementation

| Contributor | Work used in this result |
|---|---|
| [OpenAI/math](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb) | The original seven-eighths initialization, Hecke and Dirichlet definitions, and analytic and arithmetic framework. |
| [Akash Levy](https://github.com/GTimG/QuasiRiemannTracker/pull/6) | Weighted-numerator work: numerator inversion, conductor allocation, coefficient estimates and contour assembly. |
| [Earlier QRH work directed by Hailey Collet](https://github.com/GTimG/QuasiRiemannTracker/pull/9) | AI-assisted reflection, variable moments, physical-source and common-profile machinery, building on the earlier [single-prime development](https://github.com/GTimG/QuasiRiemannTracker/pull/8). |
| [Zhongpai Gao](https://github.com/GTimG/QuasiRiemannTracker/pull/15) | AI-assisted local moments, Fourier/source reconstruction and physical bridges used in the local refinement. |
| [nielstron](https://github.com/GTimG/QuasiRiemannTracker/pull/26) | The dependency-ordered bundling approach with isolated sections that informed the submission layout. |

The [pinned source-attribution record](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/fb6d4fd11aeaff0b37dd96bad3b274ce28a5c08b/proofs/cycle61-c0125139/PROVENANCE.md) gives the exact revisions and used modules. Mathlib and the other dependencies, as well as Lean, Comparator, NanoDa, con-ron and their supporting tools, retain their own authorship and licenses. The paper also identifies the primary literature behind the imported mathematics.

The manuscript and new code are available under Apache-2.0. Reused work retains its original notices and licenses. Historical notices credit the research packages as they were supplied; the authorship account above explains the human and AI roles within my contribution.
