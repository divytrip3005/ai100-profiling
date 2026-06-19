# MatMul Execution — Precise Walkthrough (seq=128)

**Shape:** `[128 × 4096] × [4096 × 11008]`, 16 cores, mxfp6=True  
**Source:** `results/h4096_o11008_s128/perf_dump/opstats/*.trace.json`

---

## Before anything starts

```
Input  [128 × 4096]   fp16    1 MB   → in DDR
Weight [4096 × 11008] mxfp6  33.6 MB → in DDR
Output [128 × 11008]  fp16    2.75 MB → will go to DDR
```

**FLOPs:** 2 × 128 × 4096 × 11008 = **11.5 billion**

---

## Key difference from seq=512

| | seq=128 | seq=512 |
|--|---------|---------|
| Input fits in VTCM? | **YES** — 1 MB fits | No — 4 MB needs chunking |
| Input DMA loads per core | **0** | 1 |
| Input loaded how | **stays in VTCM from host** | chunked from DDR |
| NumSrcOperands | **4** (input K split into 4 slices of 1024) | 1 (full K in one slice) |
| HMX ops per core | **22** | 43 |
| Bottleneck | **BANDWIDTH-BOUND** | COMPUTE-BOUND |

The input is small enough (1 MB) that it fits entirely in VTCM and is
**never loaded from DDR by the cores** — it arrives once from the host
and stays resident throughout.

---

## Core assignment — N-split

```
11008 output channels ÷ 16 cores = 688 channels per core
HMX processes 32 channels per call → 688/32 = 21.5 → 22 HMX ops per core
22 × 16 = 352 total HMX ops
```

Source: `aicconvolutiond32` count per core from trace = 22 each.

> Note: seq=512 needed 43 ops/core because it had 2 input passes (512/256=2).
> seq=128 has OutYEnd=128 = full M in one pass → only 22 ops/core.

---

## Step 1 — Input already in VTCM (no DMA needed)

**0 input DMA loads** observed across all 16 cores.

The input `[128 × 4096]` = 1 MB is loaded directly into every core's VTCM
by the host before inference starts — in flat fp16 format.

```
aicinputsemaphoreinc on core 0: x/__19  [1×128×4096] = 1024KB  TCM
aicinputsemaphoreinc on core 1: x/__20  [1×128×4096] = 1024KB  TCM
...  (every core has its own copy in TCM)
```

Source: `aicinputsemaphoreinc` events per core — all show `memory=TCM`.

---

## Step 1b — VTCM-to-VTCM copy (staging buffer)

Each core makes an internal VTCM copy of the input into a working buffer:

```
Core 0 ts=4µs: aiccopysamevtcm  x/__19 → x/__52  [128×4096]  TCM → TCM
Core 1 ts=4µs: aiccopysamevtcm  x/__20 → x/__55  [128×4096]  TCM → TCM
...
```

This creates a working buffer before D32 conversion.
Source: `aiccopysamevtcm` events at `ts=4µs` per core.

---

## Step 2 — D32 conversion: 32 rows × 1/4 K per core

Each core converts its assigned slice of the input to D32 layout.
The slice is **32 rows × 1024 K** (not the full K=4096):

```
Input  (flat fp16): [1×32×4096]       = 256KB
                     ↓  DPadEnd=1024 (only 1024 K elements processed)
Output (D32 layout): [1×1×1×32×1024]  =  64KB

YPadEnd = 32    → 32 rows
DPadEnd = 1024  → 1024 K elements (K/4)
```

The 16 cores are assigned different (row, K-slice) combinations:

```
Core  0: rows  0–31  × K    0–1023  → D32 → [64KB]
Core  1: rows  0–31  × K 1024–2047  → D32 → [64KB]
Core  2: rows  0–31  × K 2048–3071  → D32 → [64KB]
Core  3: rows  0–31  × K 3072–4095  → D32 → [64KB]
Core  4: rows 32–63  × K    0–1023  → D32 → [64KB]
Core  5: rows 32–63  × K 1024–2047  → D32 → [64KB]
Core  6: rows 32–63  × K 2048–3071  → D32 → [64KB]
Core  7: rows 32–63  × K 3072–4095  → D32 → [64KB]
Core  8: rows 64–95  × K    0–1023  → D32 → [64KB]
Core  9: rows 64–95  × K 1024–2047  → D32 → [64KB]
Core 10: rows 64–95  × K 2048–3071  → D32 → [64KB]
Core 11: rows 64–95  × K 3072–4095  → D32 → [64KB]
Core 12: rows 96–127 × K    0–1023  → D32 → [64KB]
Core 13: rows 96–127 × K 1024–2047  → D32 → [64KB]
Core 14: rows 96–127 × K 2048–3071  → D32 → [64KB]
Core 15: rows 96–127 × K 3072–4095  → D32 → [64KB]
```

16 cores × (32 rows × 1024 K) = 4 row-groups × 4 K-groups = full [128×4096] covered.

Source: `aicconverttod32` — 1 event per core, `opAttributes: YPadEnd=32, DPadEnd=1024`.

---

## Step 3 — Input multicast: each core shares its 64KB slice

Each core multicasts its D32 slice (64KB) to all 15 other cores:

```
Core  0 sends: rows  0–31  × K    0–1023  in D32  → all 15 other cores
Core  1 sends: rows  0–31  × K 1024–2047  in D32  → all 15 other cores
...
Core 15 sends: rows 96–127 × K 3072–4095  in D32  → all 15 other cores
```

After all 16 multicasts complete, every core has the full input
assembled as 4 K-slices (NumSrcOperands=4):

```
op3: [1×1×4×32×1024] = rows 0–127  × K    0–1023  (4 row-chunks of 32 rows)
op4: [1×1×4×32×1024] = rows 0–127  × K 1024–2047
op5: [1×1×4×32×1024] = rows 0–127  × K 2048–3071
op6: [1×1×4×32×1024] = rows 0–127  × K 3072–4095
```

HMX concatenates op3–op6 along K (`InputConcatDim=3`) → full `[128×4096]`.

Source:
- `aicmulticastvtcm` — 1 event per core, shape `<1×1×1×32×1024>=64KB`
- D32 output tensor id matches multicast source id (`__7253` on core 0) ✓

> **Why multicast if input is already in every core's VTCM?**
> The input exists in VTCM as flat fp16. HMX cannot use flat fp16 —
> it requires D32 layout. Each core only D32-converts its own 64KB slice
> (32 rows × 1024 K). The multicast shares these D32 slices so every
> core ends up with the full input in D32 format.



---

## Step 4 — Weight tile #1 loads

At `ts=33µs`, DMA copies first weight tile from DDR → VTCM:

```
Shape (mxfp6): [1×1×102400] = 100 KB
Covers: 4096 K × 32 output channels (compressed)
Duration: 16.8 µs
```

**22 weight loads per core** — each core loads all its weight tiles from DDR.
Total = 22 × 16 = 352, but N/32 = 344... slight over-count due to padding.

Source: 22 `aiccopytovtcm` events per core with shape `<1×1×102400>=100KB`.

---

## Step 5 — HVX decompresses weight

At `ts=50µs`, HVX expands weight tile:

```
Input:  [1×1×102400] = 100 KB  mxfp6
Output: [1×1×131072] = 256 KB  fp16  (actual: split into 4 sub-tiles of 64KB each)
Duration: 3.9 µs
```

Source: `blockdequantize_mxfp6` event, 22 per core.

---

## Step 6 — HMX computes one tile

At `ts=56µs` (only **2µs after dequant finishes**), HMX fires:

```
Input  (op3–op6): 4 × [1×1×4×32×1024] = 4 × 256KB = 1024KB total
                  → 128 rows × 4096 K  split into 4 sub-tiles (NumSrcOperands=4)
Weight (op4):     [1×1×4×32×1024]     = 256 KB
                  → 4096 K × 32 outC  split into 4 sub-tiles
Output (op0):     [1×1×4×8×1024]      = 64 KB
                  → 128 rows × 32 outC

OutYEnd        = 128   rows  (full M — all rows in one pass)
OutCPerG       = 32    output channels
InCPerG        = 4096  full K
NumSrcOperands = 4     (K split into 4 sub-tiles of 1024 each)
Duration       = 16.2 µs
```

**Why NumSrcOperands=4 here but 1 in seq=512?**

Formula: `NumSrcOp = max(1, ceil(cores / ceil(M/32)))`  
= `max(1, ceil(16 / ceil(128/32)))` = `max(1, ceil(16/4))` = **4**

The K dimension (4096) is split into 4 sub-tiles of 1024 each because M=128
is small — the compiler needs to split K to keep HMX busy.

Source: `aicconvolutiond32` op attributes from trace.

---

## Step 7 — Tight pipeline: DMA → DEQUANT → HMX back-to-back

The pipeline is much tighter than seq=512:

```
ts= 33µs : DMA  WEIGHT #1   100KB   dur=16.8µs
ts= 50µs : DEQUANT #1       → 256KB dur=3.9µs   ← starts as soon as #1 arrives
ts= 50µs : DMA  WEIGHT #2   100KB   dur=16.3µs  ← overlaps with dequant
ts= 56µs : HMX  #1 fires    dur=16.2µs          ← 2µs after dequant done
ts= 67µs : DEQUANT #2       → 256KB dur=10.8µs
ts= 67µs : DMA  WEIGHT #3   100KB   dur=19.7µs
ts= 78µs : HMX  #2 fires    dur=17.9µs
ts= 88µs : DEQUANT #3       → 256KB
ts= 88µs : DMA  WEIGHT #4   100KB
ts=102µs : HMX  #3 fires
...
```

**Prefetch depth = 1** — HMX fires only 2µs after tile #1 is ready.
No deep prefetch because the input is already in VTCM — no input setup
time to fill with prefetched weight tiles.

Each cycle: `DMA load (~17µs) → DEQUANT (~10µs) → HMX (~15µs)`
— these three steps run in a tight overlapping loop for all 22 tiles.

---

## Step 8 — Output written to DDR

After all 22 HMX ops, output is converted and written:

```
Per HMX call:  [128 × 32]  = 64 KB output tile
Per core total: 128 rows × 688 channels × 2 bytes = 176 KB → DDR
All 16 cores:  128 rows × 11008 channels × 2 bytes = 2.75 MB → DDR
```

> Notice from DDR traffic log: output (`QAicGraph_//MatMul/`) shows
> **0 bytes** in Non-Const Out — output stays in VTCM and is returned
> directly to host. The output (2.75 MB) fits in VTCM for this shape.

Source: `QAicGraph__ddr_op_summary.log` — DDR Out shows only profiling data.

---

## VTCM at peak (measured)

```
Input tile    128 rows × 4096 K × 2 bytes   =  1024 KB  (full input resident)
Weight tile   4096 K   × 32 outC × 2 bytes  =   256 KB
Output tile   128 rows × 32 outC × 2 bytes  =    64 KB
HMX scratch   fixed                          =    68 KB
────────────────────────────────────────────────────────
Formula total                               =  1412 KB

Compiler overhead (double-buffer, D32 format
buffers, staging)                           =  5992 KB
────────────────────────────────────────────────────────
Actual peak measured                        =  7404 KB
Utilisation                                 =  90.4% of 8192 KB
```

Source: `QAicGraph__memuse_estimate_final.log` VTCM highwatermark.

> VTCM utilisation is **90.4%** — much higher than seq=512 (64.1%).
> The full input (1024 KB) staying resident throughout contributes
> significantly to the higher utilisation.

---

## Time breakdown (wall clock = 504 µs)

| Operation | % | Total µs (all cores) | What it is |
|-----------|---|----------------------|------------|
| sync_stall | 63.9% | 10,004 | HVX waiting for HMX + DDR latency |
| actual_compute | 18.5% | 2,893 | HMX multiply-accumulate |
| weight_dequant | 11.4% | 1,779 | HVX mxfp6 → fp16 |
| data_movement | 4.3% | 676 | DMA loading weight tiles |
| format_convert | 1.3% | 202 | D32 layout reformatting |

**Sync stall dominates at 63.9%** — much worse than seq=512 (37%).
HMX finishes each tile quickly but then waits for the next weight tile
to arrive from DDR and be dequanted. The pipeline is DDR-bound.

---

## Bottleneck — BANDWIDTH-BOUND

```
Arithmetic intensity = FLOPs / DDR bytes
= 11.5B / 33.6MB = 341 FLOPs/byte

DDR bandwidth observed = 65.1 GB/s
BW-bound minimum time  = 33.6MB / 65.1GB/s = 516 µs
Actual wall clock      = 504 µs
HMX active             = 35.5%
```

AI = 341 FLOPs/byte is much lower than seq=512 (916 FLOPs/byte).
Only 33.6 MB of weight loaded but only 11.5B FLOPs to compute —
HMX is idle 64.5% of the time waiting for DDR.

---

## Comparison: seq=128 vs seq=512

| Aspect | seq=128 | seq=512 |
|--------|---------|---------|
| Input location | **VTCM** (resident) | DDR (chunked) |
| Input DMA loads | **0** | 16 |
| NumSrcOperands | **4** (K split) | 1 (full K) |
| HMX ops/core | **22** | 43 |
| OutYEnd | **128** (1 pass) | 256 (2 passes) |
| Prefetch depth | **1 tile** | 4 tiles |
| VTCM peak | **90.4%** | 64.1% |
| Wall clock | **504 µs** | 1029 µs |
| HMX% | **35.5%** | 70.0% |
| Sync stall% | **63.9%** | 37.0% |
| Bottleneck | **BANDWIDTH** | COMPUTE |
| DDR traffic | **33.6 MB** (weights only) | 48.4 MB |

Key insight: at seq=128, the entire 11008-channel output for 128 rows
needs only 11.5B FLOPs but still requires loading the full 33.6 MB weight
matrix. Each byte of weight is only reused 341 times vs 916 times at seq=512.
**Less reuse = more bandwidth pressure = bandwidth-bound.**

---

## Verified numbers

| What | Exact value | Source |
|------|-------------|--------|
| Input DMA loads | 0 | trace `aiccopytovtcm` count |
| Input location | VTCM (resident from host) | trace — no DDR src x/ events |
| Input multicast shape | `[1×1×1×32×1024]` = 64 KB | trace `aicmulticastvtcm` |
| Weight DMA shape | `[1×1×102400]` = 100 KB mxfp6 | trace `aiccopytovtcm` |
| Weight after dequant | `[1×1×131072]` = 256 KB fp16 | trace `blockdequantize_mxfp6` |
| Weight loads per core | 22 | trace count |
| HMX input operand | `[1×1×4×32×1024]` × 4 = 256 KB each | trace op3–op6 shape |
| HMX weight operand | `[1×1×4×32×1024]` = 256 KB | trace op4 shape |
| HMX output operand | `[1×1×4×8×1024]` = 64 KB | trace op0 shape |
| HMX OutYEnd | 128 rows (full M, 1 pass) | trace op attributes |
| HMX OutCPerG | 32 channels | trace op attributes |
| HMX InCPerG | 4096 | trace op attributes |
| HMX NumSrcOperands | **4** (K split into 4 × 1024) | trace op attributes |
| HMX ops per core | 22 | trace count |
| Prefetch depth | 1 tile | trace timestamps |
| Delay after tile ready | 2 µs | trace timestamps |
| VTCM peak | 7404 KB (90.4%) | memuse log |
| Wall clock | 504 µs | raw_device_stats |
| HMX utilisation | 35.5% | raw_device_stats |
| Core imbalance | 10.7% | trace `Core-N-Execution` |

---

## Files referenced

```
results/h4096_o11008_s128/
└── perf_dump/
    ├── dumps/
    │   ├── QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log
    │   ├── QAicGraph__op_summary_final.log
    │   ├── QAicGraph__memuse_estimate_final.log
    │   └── QAicGraph__ddr_op_summary.log
    └── opstats/
        └── *.qaic-opstats.trace.json
```
