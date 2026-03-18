---
name: Always run all test suites before validating changes
description: Run both targetDetection.test.ts and groundTruth.test.ts before declaring a change correct
type: feedback
---

Always run both test suites before validating any code change:

```
npm test -- --testPathPattern=targetDetection
npm test -- --testPathPattern=groundTruth
```

**Why:** Speed optimizations (e.g. BOOTSTRAP_SCALE=4 downsampling) passed targetDetection.test.ts but regressed groundTruth.test.ts — calibration sanity checks and boundary accuracy tests exposed issues that the structural tests missed.

**How to apply:** After any change to targetDetection.ts, run both suites. groundTruth.test.ts has one pre-existing failure (20190325_202607.jpg boundary corner at 142px vs 60px tolerance) — this is expected and not a regression.
