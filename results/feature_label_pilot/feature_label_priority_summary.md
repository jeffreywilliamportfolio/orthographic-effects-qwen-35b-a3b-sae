# Feature Label Priority Summary

Evidence-only ranking from the two-prompt feature-label pilot. No semantic labels are assigned here.

## Ranking Method

Features are prioritized by condition-specific TopK-50 presence first, then absolute activation delta, then rank movement for features present in both conditions. Binary-only patterns are ranked ahead of shifted-in-both patterns because they provide cleaner first candidates for high-activation example collection.

## Pattern Counts

- `dstroke_only`: 2
- `normal_only`: 7
- `both_shifted`: 1
- `both_stable`: 0
- `weak_or_unclear`: 0

## First Candidates

1. Layer 26 feature 23977 (`normal_only`): Feature appears only in normal TopK-50; abs_delta=0.192175.
2. Layer 26 feature 2722 (`normal_only`): Feature appears only in normal TopK-50; abs_delta=0.18328.
3. Layer 26 feature 9745 (`normal_only`): Feature appears only in normal TopK-50; abs_delta=0.173812.
4. Layer 26 feature 7108 (`normal_only`): Feature appears only in normal TopK-50; abs_delta=0.17277.
5. Layer 26 feature 31784 (`dstroke_only`): Feature appears only in d-stroke TopK-50; abs_delta=0.170044.

## Full Ranked List

| Priority | Layer | Feature | Pattern | Normal Activation | D-stroke Activation | Delta | Normal Rank | D-stroke Rank | Evidence |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 26 | 23977 | `normal_only` | 0.192175 | 0 | -0.192175 | 30 |  | Feature appears only in normal TopK-50; abs_delta=0.192175. |
| 2 | 26 | 2722 | `normal_only` | 0.18328 | 0 | -0.18328 | 35 |  | Feature appears only in normal TopK-50; abs_delta=0.18328. |
| 3 | 26 | 9745 | `normal_only` | 0.173812 | 0 | -0.173812 | 41 |  | Feature appears only in normal TopK-50; abs_delta=0.173812. |
| 4 | 26 | 7108 | `normal_only` | 0.17277 | 0 | -0.17277 | 43 |  | Feature appears only in normal TopK-50; abs_delta=0.17277. |
| 5 | 26 | 31784 | `dstroke_only` | 0 | 0.170044 | 0.170044 |  | 40 | Feature appears only in d-stroke TopK-50; abs_delta=0.170044. |
| 6 | 14 | 30172 | `normal_only` | 0.127116 | 0 | -0.127116 | 39 |  | Feature appears only in normal TopK-50; abs_delta=0.127116. |
| 7 | 14 | 12433 | `normal_only` | 0.119017 | 0 | -0.119017 | 46 |  | Feature appears only in normal TopK-50; abs_delta=0.119017. |
| 8 | 14 | 3291 | `dstroke_only` | 0 | 0.118389 | 0.118389 |  | 41 | Feature appears only in d-stroke TopK-50; abs_delta=0.118389. |
| 9 | 14 | 28025 | `normal_only` | 0.115953 | 0 | -0.115953 | 47 |  | Feature appears only in normal TopK-50; abs_delta=0.115953. |
| 10 | 14 | 9030 | `both_shifted` | 0.137331 | 0.109591 | -0.0277399 | 29 | 47 | Feature appears in both TopK-50 but shifts between conditions; abs_delta=0.0277399; rank_shift=18. |

## Restrictions Confirmed

No model run, seed bank, steering, Hauhau, llama.cpp, full experiment, or semantic labels were used.
