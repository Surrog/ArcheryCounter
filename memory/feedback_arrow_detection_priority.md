---
name: Arrow detection priority — precision over recall
description: User explicitly prefers missing arrows over false positives in arrow detection
type: feedback
---

Precision is the priority for arrow detection. Missing a real arrow (false negative) is acceptable; detecting a spurious one (false positive) is not.

**Why:** User wants to trust every detection that is shown. A missed arrow is a known gap; a false positive is misleading and undermines confidence in the whole result.

**How to apply:** When tuning arrow detection thresholds, filters, or suggesting fixes, always bias toward fewer detections rather than more. Prefer conservative thresholds, tighter filters, and early rejection. Do not optimise for recall at the expense of precision.
