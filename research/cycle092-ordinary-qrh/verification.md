---
layout: page
title: "Cycle092 local verification record"
permalink: /research/cycle092-ordinary-qrh/verification/
---

[Back to Cycle092](../)

## Initial development check

The source-bound local receipt is dated 2026-11-20T04:06:57.941023+00:00. It reports successful compilation of the unconditional final root with Lean 4.34.1, using the development dependency objects. A separate process loaded that root, checked the exact all-Dirichlet signature and audited both exported declarations:

- `QRH.Cycle092.dirichlet_nonvanishing`
- `QRH.Cycle092.hecke_and_dirichlet_record`

Both report only `propext`, `Classical.choice` and `Quot.sound`. No extra analytic premise occurs in the checked final all-Dirichlet signature.

The final-root source SHA-256 is `005e7ae5a54853586d9d591ce45814ec4ca023191ca5fafc935eedd0509711cd`. The [machine-readable record](../cycle092-local-lean-50a933d719e4.json) retains the compiler, root and source-archive identities. The downloadable Lean source is the committed submission at `50a933d719e4adf2775fabe9774a53000dfea522`; its archive SHA-256 is `193b83c75a035232ef2060b45d8b2ad3138d415b37beaa0d577e3d27c89189b0`. An archive identity is not a proof-replay receipt.

The initial development receipt does **not** establish a complete clean submission build, canonical independent kernel replay, GTimG’s protected exact-commit verification, or tracker admission. Those outcomes require their own exact-revision evidence.

## Later staged Candidate-source check

The [separate staged-source and axiom receipt](../cycle092-staged-lean-50a933d719e4.json) is bound to the same submission commit `50a933d719e4adf2775fabe9774a53000dfea522`. It reports PASS for 2,325 Candidate source modules using 4,405 approved dependency objects, zero module failures, a successful canonical-signature audit, and only `propext`, `Classical.choice` and `Quot.sound`. The run retained 245 successful first-phase modules after a controlled scheduler resume and compiled the remaining 2,080 modules. The dependency objects were reused under the recorded checks.

This is a fresh Candidate-source check against approved dependencies. It does not claim that every upstream dependency source was freshly compiled, and it does not establish canonical independent kernel replay, protected GTimG verification or tracker admission. The initial development receipt above remains unchanged.


The formal Lean submission is [GTimG PR #144](https://github.com/GTimG/QuasiRiemannTracker/pull/144). Its protected checks and tracker admission remain separate from these local receipts.

The [written proof PR #45](https://github.com/rohanarun/quasi-rh-hillclimb/pull/45) belongs to the separate written-proof repository, `rohanarun/quasi-rh-hillclimb`.
