---
layout: page
title: Contributions and attribution
permalink: /research/cycle61-qrh-boundary/contributions/
---

**Hailey Collet** · [HaileyCollet@gmail.com](mailto:HaileyCollet@gmail.com)

Updated 13 October 2026

[Back to the paper and downloads](/research/cycle61-qrh-boundary/)

I initiated and directed the QRH research program that led to the selected **c = 0.125139** written proof. My work included setting the objectives, originating research questions, choosing sources and proof branches, challenging dependencies, and organizing comparative research and integration. I am the sole human author of the resulting manuscript and formalization package.

The detailed analytic derivations were produced with AI assistance. **Astra and GPT-6.1 Sol performed the Lean formalization.** I commissioned and coordinated that work and transferred its results back into the continuing research; I did not personally formalize the proof. Reused mathematics retains its original authorship, notices and licenses.

## The original written-proof handoff

The selected written-proof packet was prepared on **11 October 2026 at 9:47 p.m. MDT (12 October, 03:47 UTC)**. That preparation point separates the research leading to the packet from the subsequent formalization and continuing program. It is not a claimed start time for the formalizer.

The contributions below describe both decisions that shaped the selected proof and exploratory directions investigated along the way. An idea's inclusion in the research agenda does not mean it became an ingredient of the final proof.

## My contributions before the handoff

- **Choosing the mathematical objective.** I began by asking for a shorter, simpler, self-contained argument, specializing to Riemann zeta where useful while increasing the zero-free constant. I set targets including 1/8 and 1/6, and made concision a lower priority after the reported crossing of 1/8. I separately selected the concise 4/33 branch for mathematical communication.
- **Challenging the dependency standard.** I questioned whether the claimed standalone proof still required earlier zero-free papers, asked for the specialized intermediate arguments to be supplied, and distinguished those obligations from general foundations that could reasonably be cited. I identified and pressed the dependency question; the missing analytic derivations were model work.
- **Proposing alternative geometries.** I suggested arguments starting from the left of the critical strip, interior bands and horizontal strips, direct confinement around the critical line, and regions approaching or touching it. My examples included the band between 2/3 and 1/√2. These were mathematical questions and proposed search directions.
- **Proposing bridges from tilings to arithmetic.** I introduced plane tilings and Einstein tiles, then suggested tori, metric topology, ergodicity, homographies and covering projections as possible connections to the proof. The resulting conditional adapters and obstructions were developed by the models; a proved arithmetic saving from those tilings was not established.
- **Directing small-block searches.** I proposed systematic grouping of a few Möbius blocks, dynamic programming and optimization methods such as simulated annealing, stochastic hill climbing and basin hopping. The ensuing restricted estimates and counterexamples to simple state compression were model-derived. My contribution was the specific search proposal and its continuation.
- **Designing comparative and creative research.** I introduced externally sourced creativity methods, including mathematical-object generation through randomized Akinator-style search. I commissioned comparable brainstorming and forward-research streams from common inputs, with separate criticism and synthesis. I required a coherent reconciled argument and a separate research agenda, rather than treating a collection of responses as completed integration. The external methods retain their original credit.
- **Selecting sources and usable proof branches.** I supplied and expanded the source material, directed comparisons with the current argument, and sought reuse of earlier formalizations. I brought Gao's local refinement into the program and required its assumptions and source lineage to remain explicit. I coordinated the selection and preparation of the written handoff for formalization.
- **Preserving continuity.** I required portable handoffs, retention of useful failures and unfinished approaches, and enough context to avoid repeating earlier investigations. I directed effort toward stronger results while preserving shorter proof branches for communication.

Some research prompts were written by assistants and then selected, assembled or transmitted by me. That selection is part of my research direction; it does not make every embedded formula a human-originated proposal. I also did not personally derive or write the earlier 4/33 proof.

## Formalization after the handoff

Astra and GPT-6.1 Sol carried out the detailed Lean development and the mathematical work needed to complete the formal assembly. My role was commissioning, coordination, source transfer and directing subsequent comparison with the continuing research. Concrete formalization-stage parameter choices and auxiliary-theorem changes belong to that model work.

The supplied written proof already contained the enlarged density range and numerical endpoint certificate. The completed development establishes the unconditional endpoint at **c = 125139/1000000**, with the strict zero-free boundary **874861/1000000** and the principal-pole exception. The final paper explains the source normalization, order of choices and complete error estimate used by the Lean construction.

The [verification record](https://github.com/HaileyStorm/QuasiRiemannTracker/blob/fb6d4fd11aeaff0b37dd96bad3b274ce28a5c08b/proofs/cycle61-c0125139/Verification.md) identifies the actual compilation, statement-comparison and kernel checks, including the limits of the adapted local replay environment. The [submission PR](https://github.com/GTimG/QuasiRiemannTracker/pull/42) reports the separate protected tracker check. Attribution and AI review do not substitute for those checks.

## Continuing research and collaboration

After the original packet was prepared, I continued directing the program. My contribution account records these further decisions on 13 October:

- **Focusing on signed cancellation.** I challenged whether the research was directly attacking the stubborn signed remainder, supplied Mertens-related references, suggested Selberg sieve weights and Brun's pure sieve, and brought in the separate Hexagon paper for assessment. Applicability, ensuing reductions and remaining estimates were assessed by the research agents; this intervention is not a claim of a larger proved constant.
- **Returning the formalized result to research.** I supplied the completed formalization package and its comparison with the original handoff, asking the research effort to identify anything useful for its continuing work.
- **Designing a shared component catalogue.** I proposed combining the local attempt index and upstream library, with searchable fields, tags, linked browsing, related and superseded components, and support for reusable mathematical and Lean material. I assigned initial Lean population to the formalization effort.
- **Defining public and private usage records.** I asked for public paper and submission links on both mathematical and formal components, with internal research-cycle usage kept in a private index. I requested contribution and citation guidance for humans and agents and regular synchronization of the shared catalogue.
- **Setting collaboration and licensing choices.** I proposed a separate public catalogue repository linked into private QRH, selected MIT for new catalogue tools and CC BY 4.0 for original catalogue descriptions, and required imported material to retain its existing terms. These catalogue choices are separate from the Apache-2.0 license of this paper and its new proof code.
- **Designing a review workflow.** I proposed autonomous mathematical and technical PR review, local workers with offline catch-up, inexpensive Jev preliminary checks, a first substantive review by GPT-6.1 Sol at High, and Astra at xhigh when the mathematics warrants escalation. Human involvement would remain available for nontechnical decisions. Software implementation and deployment are agent work; this account records the design decisions without claiming the workflow had already reviewed an external PR.

These later contributions belong to the continuing research and collaboration program. They are not retroactive contributions to the mathematical content of the original c = 0.125139 packet.

## AI assistance across the program

The research account records assistance from GPT6-Pro, non-Pro Astra, GPT-6.1 Sol, Grok 4.7 and Gemini 3.1 Pro. The explicit attribution for the Lean formalization is Astra and GPT-6.1 Sol. These names identify tools used across the program, rather than assigning every lemma to a model without supporting records.

This page adapts my updated contribution account of 13 October. Its detailed chronology and private working references are retained in the research record. The account covers the original research, selected later source integration, formalization and the later decisions described above; it does not assign a distinct human intervention to every numbered research cycle.

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
