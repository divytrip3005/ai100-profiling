# OutCPerG=256 Cliff — h4096 × o11008, 16 cores, mxfp6=True

**Shape:** `[seq × 4096] × [4096 × 11008]`  
**Config:** 16 cores, mxfp6=True, 1 device  
**Source:** `opstats/*.trace.json`, `blockdequantize_mxfp6` events, `aicconvolutiond32` opAttributes

---

## The cliff — seq=960–1024 performs 4× worse than neighbours

| seq | ExecTime | TFLOPS | HMX% | DDR MB | OutCPerG | Dequant pattern |
|-----|---------|--------|------|--------|----------|-----------------|
| 768 | 1,681 µs | 41.2 | 76.1% | 55.8 | 32 | TCM→TCM ✓ |
| 832 | 1,959 µs | 38.3 | 74.2% | 57.7 | 32 | TCM→TCM ✓ |
| 896 | 2,331 µs | 34.7 | 80.3% | 59.5 | 32 | TCM→TCM ✓ |
| **960** | **10,649 µs** | **8.1** | **16.1%** | **663** | **256** | **DDR→DDR ⚠** |
| **992** | **10,687 µs** | **8.4** | **16.3%** | **662** | **256** | **DDR→DDR ⚠** |
| **1024** | **10,691 µs** | **8.6** | **17.1%** | **665** | **256** | **DDR→DDR ⚠** |
| 1152 | 3,066 µs | 33.9 | 74.7% | 66.7 | 32 | TCM→TCM ✓ |
| 1280 | 3,414 µs | 33.8 | 81.9% | 70.6 | 32 | TCM→TCM ✓ |
| 1536 | 4,981 µs | 27.8 | 69.7% | 78.3 | 32 | TCM→TCM ✓ |

The cliff affects only seq=960, 992, 1024. Both neighbours (896 and 1152) work correctly.

---

## Why seq=512–1536 (excl 960–1024) work well

All good seq_lens share the same clean pipeline:

```
OutCPerG = 32  (small output tile per HMX op)
Weight tile = 100KB mxfp6  →  loaded tile by tile from DDR → VTCM
Dequant = TCM→TCM  (each core dequants its own tile in its own VTCM)
All 16 cores work in parallel throughout execution
```

Per weight tile on every core:
```
DMA: load 100KB mxfp6 from DDR → VTCM  (~17µs)
HVX: dequant 100KB→256KB fp16 in VTCM  (~4µs)
HMX: compute [OutY × 4096] × [4096 × 32]  (~15µs)
Discard → next tile → repeat
```

No serialisation, no DDR writeback, all 16 cores busy simultaneously.

---

## Why seq=960–1024 perform so badly

At seq=960 the compiler switches `OutCPerG` from 32 → **256** (8× larger).
This triggers a cascade:

```
OutCPerG = 256
    ↓
Weight tile = [4096 × 256] fp16 = 2048KB  (was 100KB mxfp6 — 20× larger)
    ↓
Compiler pre-dequants ENTIRE weight matrix at once:
  33.6MB mxfp6 → 88MB fp16 → written to DDR
    ↓
Single dequant on core 0 only → takes 5,290µs
15 other cores sit IDLE during this 5.3ms
    ↓
All cores load large 2048KB fp16 tiles from DDR
DDR traffic: 665MB vs expected ~60MB (11× excess)
```

Verified from trace:
```
seq=960–1024: blockdequantize_mxfp6
  count   = 1  (core 0 only)
  IN      = [DDR] 34,400KB  (full weight matrix, mxfp6)
  OUT     = [DDR] 88,064KB  (full weight matrix, fp16)
  duration = 5,290µs

seq=896, 1152: blockdequantize_mxfp6
  count   = 21 per core × 16 cores = 336 total (all parallel)
  IN      = [TCM] 100KB  (one tile)
  OUT     = [TCM] 256KB  (one tile)
  duration = ~4µs per tile
```

---

## What seq=960–1024 SHOULD look like (ideal)

If the compiler had kept OutCPerG=32 (like seq=896 and seq=1152),
interpolating from neighbours:

| seq | Actual | Should be | Wasted time | Slowdown |
|-----|--------|-----------|-------------|----------|
| 960 | 10,649 µs (8.1 TFLOPS) | **2,515 µs (34.4 TFLOPS)** | 8,134 µs | **4.2×** |
| 992 | 10,687 µs (8.4 TFLOPS) | **2,607 µs (34.3 TFLOPS)** | 8,081 µs | **4.1×** |
| 1024 | 10,691 µs (8.6 TFLOPS) | **2,698 µs (34.2 TFLOPS)** | 7,992 µs | **4.0×** |

This is a **compiler tiling bug** — no hardware limitation causes this.

---

## Three fixes

### Fix 1 — Disable mxfp6 *(verified)*

```bash
python matmul_microbenchmark.py --seq-len 1024 --no-mxfp6 ...
```

Without mxfp6 the compiler has no dequant node → goes back to OutCPerG=32 → clean pipeline.

```
seq=1024 mxfp6=True:  10,885 µs   8.5 TFLOPS  ← cliff
seq=1024 mxfp6=False:  3,553 µs  26.0 TFLOPS  ← 3× faster
```

Source: `results/h4096_o11008_s1024_nofp6/`  
**Downside:** weight 2.7× larger in memory (86MB vs 32MB)

---

### Fix 2 — Bucket to nearest good seq_len *(estimated)*

Compile for seq=896 (last good) or seq=1152 (first good after cliff).
Pad input rows with zeros at runtime.

```bash
# Option A: compile for seq=896
python matmul_microbenchmark.py --seq-len 896 ...

# Option B: compile for seq=1152
python matmul_microbenchmark.py --seq-len 1152 ...
```

Expected: ~2,515–2,700 µs → **4× faster than cliff**  
**Downside:** slight compute waste on padding rows (56–128 extra rows out of 960–1024)

---

### Fix 3 — Two seq=512 calls *(estimated best)*

Split `[1024 × 4096]` into two `[512 × 4096]` calls:

```
Call 1: seq=512, rows 0–511    → 1,029 µs  (COMPUTE-BOUND, 70% HMX)
Call 2: seq=512, rows 512–1023 → 1,029 µs
─────────────────────────────────────────
Total                          ≈ 2,058 µs  vs 10,685 µs  →  5× faster
```

seq=512 is the sweet spot for h4096 (COMPUTE-BOUND, 44.9 TFLOPS, 75.6% VTCM).  
**Downside:** 2 kernel launches instead of 1 (small fixed overhead each)

---

## Comparison

| Fix | Speedup | Memory cost | Complexity |
|-----|---------|-------------|------------|
| No-mxfp6 | **3×** | 2.7× weight memory | None — one flag |
| Bucket to 896/1152 | **4×** | None | Small — padding logic |
| Two seq=512 calls | **5×** | None | Small — split loop |

**Recommended:** Fix 3 (two seq=512 calls) — best speedup, no memory cost.

---

## How to detect this issue in any new shape

1. Check `blockdequantize_mxfp6` events — if count=1 and `OUT=[DDR]` → cliff
2. Check `aicconvolutiond32` opAttributes — if `OutCPerG > 32` with mxfp6 → likely cliff
3. Check DDR traffic — if >10× theoretical minimum → cliff

Theoretical minimum DDR = weight (mxfp6) + input (fp16) + output (fp16):
```
h4096 seq=1024: 32.2MB + 8MB + 22MB = 62.2MB
Actual:         665MB  →  10.7× over minimum  ← cliff confirmed
```

---

## Full seq_len map — h4096 × o11008, 16 cores, mxfp6

```
seq    ExecUs   TFLOPS  HMX%   Status
─────────────────────────────────────────────────────
1       376     0.24   13%   ⚠ too small, BW-overhead
128     505    22.9   35%   ⚠ BW-bound, low reuse
512    1029    44.9   70%   ✓ sweet spot
768    1681    41.2   76%   ✓ good
832    1959    38.3   74%   ✓ good
896    2331    34.7   80%   ✓ good
960   10649     8.1   16%   ⚠ CLIFF (OutCPerG=256 bug)
992   10687     8.4   16%   ⚠ CLIFF
1024  10691     8.6   17%   ⚠ CLIFF
1152   3066    33.9   75%   ✓ recovered
1280   3414    33.8   82%   ✓ good — best HMX%
1536   4981    27.8   70%   ✓ good
1792   8542    18.9   39%   ⚠ output spill
2048  10016    18.4   49%   ⚠ output spill
2560  14604    15.8   44%   ⚠ output spill
3072  16676    16.6   42%   ⚠ output spill
4096  25972    14.2   40%   ⚠ second cliff (OutCPerG=256)
```

---

## Files referenced

```
results/h4096_o11008_s{seq}/perf_dump/opstats/*.trace.json
results/h4096_o11008_s1024_nofp6/                          ← Fix 1 verified run
```
