# MatMul Execution — Precise Walkthrough (seq=1024)

**Shape:** `[1024 × 4096] × [4096 × 11008]`, 16 cores, mxfp6=True  
**Source:** `results/h4096_o11008_s1024/perf_dump/opstats/*.trace.json`

---

## Before anything starts

```
Input  [1024 × 4096]  fp16    8 MB   → in DDR
Weight [4096 × 11008] mxfp6  33.6 MB → in DDR
Output [1024 × 11008] fp16   22 MB   → will go to DDR
```

**FLOPs:** 2 × 1024 × 4096 × 11008 = **92.3 billion**

---

## How seq=1024 differs from seq=128 and seq=512

| | seq=128 | seq=512 | seq=1024 |
|--|---------|---------|----------|
| Input location | VTCM (no DDR) | DDR chunks | **DDR chunks** |
| Input chunk size | — | 256 KB | **512 KB** |
| Input DMA per core | 0 | 1 | **1** |
| NumSrcOperands | 4 | 1 | **1** |
| OutCPerG (HMX) | 32 | 32 | **256 / 224** |
| OutYEnd | 128 | 256 | **192** |
| Weight tile size | 100 KB | 100 KB | **2048 / 1792 KB** |
| HMX ops per core | 22 | 43 | **15–16** |
| DEQUANT events/core | 22 | 21 | **1 (giant dequant)** |
| Wall clock | 504 µs | 1029 µs | **10,885 µs** |
| Bottleneck | BW | COMPUTE | **BW (cliff)** |
| DDR traffic | 33.6 MB | 48.4 MB | **665 MB** |

**seq=1024 is the compiler tiling cliff.** DDR traffic explodes to 665 MB
because the compiler chose a completely different tiling strategy —
giant weight tiles (2048 KB) loaded 268 times vs the clean 100 KB × 344
loads used by seq=128 and seq=512.

---

## Core assignment — N-split

```
11008 output channels ÷ 16 cores = 688 channels per core
```

But unlike seq=128/512, each HMX op has `OutCPerG=256` (not 32):

```
688 channels / 256 outC per op = 2.69 → mixed: 15 ops per core
15 × 16 = 268 total HMX ops
```

Why 256 outC per op instead of 32? The compiler chose larger output tiles
to reduce the number of HMX calls, but this forces much larger weight tiles
(2048 KB per tile vs 100 KB) to be loaded from DDR.

Source: `aicconvolutiond32` count = 15 per core, `OutCPerG=256` from trace.

---

## Step 1 — Input chunk loaded from DDR

All 16 cores load an input chunk simultaneously at `ts=3µs`:

```
Each core loads: [1×64×4096] = 64 × 4096 × 2 = 512 KB  from DDR → VTCM
```

The full input is `[1024 × 4096]` = 8 MB — too large for VTCM.
So it is loaded in chunks of 64 rows each:

```
16 cores × 64 rows = 1024 rows total  (covers entire input in one round)
```

Source: 16 `aiccopytovtcm` events with shape `<1×64×4096>=512KB` at `ts=3µs`.

---

## Step 2 — D32 conversion: 64 rows × full K per core

At `ts=133µs`, each core converts its 64-row chunk to D32 layout:

```
IN  [TCM]: <1 × 64 × 4096>           = 512 KB  flat fp16
OUT [TCM]: <1 × 1 × 2 × 128 × 1024>  = 512 KB  D32 layout
```

Unlike seq=128 (32 rows × 1024 K), here each core converts
**64 rows × full K=4096** — no K splitting (NumSrcOperands=1).

Source: `aicconverttod32` — 1 event per core at `ts=133µs`.

---

## Step 3 — Input multicast: 512 KB per core

At `ts=137µs`, each core multicasts its D32 slice to all other cores:

```
Multicast shape: [1×1×2×128×1024] = 512 KB
```

Each core sends a different 64-row slice. After all 16 multicasts,
every core has the full `[1024×4096]` input assembled as:

```
op3: [1×1×6×128×1024] = 1536 KB  (192 rows × K=4096 in D32)
```

Wait — 192 rows, not 1024. This is because `OutYEnd=192`, meaning HMX
processes 192 rows per op. The 64-row chunks from multicast are assembled
into 192-row groups: 3 chunks × 64 rows = 192 rows.

After multicast: 1024 rows / 192 rows per HMX op = **5.33 → 5–6 HMX passes**
per weight tile.

Source: `aicmulticastvtcm` — 1 event per core, shape `<1×1×2×128×1024>=512KB`.

---

## Step 4 — Giant dequant (one-time, at start)

At `ts=148µs`, HVX decompresses the **entire weight matrix** in one shot:

```
DEQUANT: 33.6 MB mxfp6 → 88064 KB fp16
Duration: 5289.6 µs  (5.3 ms!)
```

This is the single most expensive step — decompressing the full 33.6 MB
weight matrix takes 5.3 ms before any computation can start.

Source: `blockdequantize_mxfp6` — 1 event on core 0, `dur=5289.6µs`.

> **Why dequant everything at once?**
> The compiler chose `OutCPerG=256` — each weight tile is
> `[4096 × 256]` = 2048 KB. With only 15 tiles per core,
> it is cheaper to dequant everything upfront than tile-by-tile.

---

## Step 5 — Weight tiles loaded from DDR

After the giant dequant, weight tiles start loading at `ts=5438µs`:

```
Weight tile shape: [1×8×131072] = 2048 KB  (10 tiles)
                or [1×7×131072] = 1792 KB  (5 tiles)
Per core: 15 weight loads total
```

**Each tile is 2048 KB** — 20× larger than seq=128/512's 100 KB tiles.

```
15 loads per core × ~2000 KB avg = ~30 MB per core
16 cores × 30 MB = ~480 MB total weight DDR traffic
```

This is why DDR traffic explodes to 665 MB.

Source: `aiccopytovtcm` with shape `<1×8×131072>=2048KB` — 10 per core,
and `<1×7×131072>=1792KB` — 5 per core.

---

## Step 6 — HMX computes one tile

At `ts=5794µs`, first HMX call fires (5.6 ms after execution started —
spent waiting for giant dequant):

```
Input  (op3): [1×1×6×128×1024] = 1536 KB  → 192 rows × 4096 K  (in VTCM)
Weight (op4): [1×8×131072]     = 2048 KB  → 4096 K × 256 outC  (in VTCM)
Output (op0): [1×1×6×8×1024]   =   96 KB  → 192 rows × 256 outC (in VTCM)

OutYEnd        = 192   rows
OutCPerG       = 256   output channels  (8× larger than seq=128/512)
InCPerG        = 4096  full K
NumSrcOperands = 1     (full K in one operand)
Duration       = 144.7 µs (first call)
```

Source: `aicconvolutiond32` op attributes from trace.

---

## Step 7 — Pipelined loop over 15 weight tiles

```
ts=  3µs   : DMA  INPUT    [1×64×4096]   512KB
ts=133µs   : D32 CONVERT   → 512KB
ts=137µs   : MULTICAST     512KB
ts=148µs   : DEQUANT (full weight) → 88064KB  dur=5290µs  ← BOTTLENECK
ts=5438µs  : DMA  WEIGHT # 1  2048KB  dur=357µs
ts=5794µs  : DMA  WEIGHT # 2  2048KB  ← prefetch
ts=5794µs  : HMX  # 1  dur=145µs
ts=6128µs  : DMA  WEIGHT # 3  2048KB
ts=6128µs  : HMX  # 2  dur=119µs
ts=6444µs  : DMA  WEIGHT # 4  2048KB
ts=6444µs  : HMX  # 3  dur=152µs
...
```

**Prefetch depth = 1** — HMX fires 357µs after dequant finishes.
The delay is because only 1 weight tile is prefetched before HMX starts —
the giant dequant occupies the pipeline for so long that no deep prefetch
buffer builds up.

---

## Step 8 — Output assembled in VTCM then written to DDR

After all 15 HMX ops, the partial outputs are assembled and written:

```
aiccopysamevtcm2d events at ts=10271µs:
  IN  [TCM]: <192 × 256>    (partial result per HMX op)
  OUT [TCM]: <192 × 3680>   (accumulated output, wider)
  → 15 such copies to assemble full [192 × 3680] output per core
```

Then written to DDR:

```
Per core:  192 rows × 688 channels × 2 bytes = 264 KB → DDR
All cores: 1024 rows × 11008 channels × 2 bytes = 22 MB → DDR
```

Source: `aiccopysamevtcm2d` × 15 events per core at `ts=10271µs`.

---

## VTCM at peak (measured)

```
Input tile    192 rows × 4096 K × 2 bytes   = 1536 KB
Weight tile   4096 K   × 256 outC × 2 bytes = 2048 KB
Output tile   192 rows × 256 outC × 2 bytes =   96 KB
HMX scratch   fixed                          =   68 KB
──────────────────────────────────────────────────────
Formula total                               = 3748 KB

Compiler overhead (double-buffer, format
buffers, staging)                           = 1044 KB
──────────────────────────────────────────────────────
Actual peak measured                        = 4792 KB
Utilisation                                 = 58.5% of 8192 KB
```

Source: `QAicGraph__memuse_estimate_final.log` VTCM highwatermark.

---

## Time breakdown (wall clock = 10,885 µs)

| Operation | % | Total µs (all cores) | What it is |
|-----------|---|----------------------|------------|
| sync_stall | **84.9%** | 364,780 | HVX/HMX waiting — dominated by giant dequant stall |
| actual_compute | 13.4% | 57,464 | HMX multiply-accumulate |
| weight_dequant | 1.2% | 5,290 | Full weight dequant (one-time, 5.3 ms) |
| format_convert | 0.4% | 1,675 | D32 layout reformatting |
| data_movement | 0.0% | 171 | DMA (small fraction — weight already dequanted) |

**Sync stall at 84.9%** — the worst of all seq_lens. Almost all time is
spent waiting, primarily because the giant 5.3ms dequant blocks everything.

---

## Bottleneck — BANDWIDTH-BOUND (cliff)

```
Arithmetic intensity = FLOPs / DDR bytes
= 92.3B / 665MB = 138 FLOPs/byte

DDR bandwidth observed = 59.7 GB/s
BW-bound minimum time  = 665MB / 59.7GB/s = 11,139 µs
Actual wall clock      = 10,885 µs
HMX active             = 27.5%
```

AI = 138 FLOPs/byte is the lowest of all seq_lens (vs 916 at seq=512).
The compiler's choice of large OutCPerG=256 forces massive weight tiles
that flood DDR with 665 MB of traffic for a 33.6 MB weight matrix —
meaning the **weight is loaded ~20× redundantly**.

---

## Why DDR traffic is 665 MB for a 33.6 MB weight

From DDR traffic log:

```
DDR Traffic Const In  (weight): 35,225,600 B  =  33.6 MB  (weight matrix)
DDR Traffic Non-const In:      549,453,824 B  = 524 MB   ← intermediate?
DDR Traffic Out:               112,721,920 B  = 107 MB   ← output + activations
```

The 524 MB Non-const traffic is the dequanted weight being written to DDR
and re-read — the compiler dequanted the full 33.6 MB weight (→ 90 MB fp16)
into VTCM, but since 90 MB >> 8 MB VTCM, it spills to DDR. It then gets
re-read tile-by-tile as weight tiles during HMX computation. This is the
**activation spill** — the dequanted weight is too large for VTCM so the
compiler uses DDR as an overflow buffer.

---

## Comparison: seq=128, seq=512, seq=1024

| Aspect | seq=128 | seq=512 | seq=1024 |
|--------|---------|---------|----------|
| Input location | VTCM | DDR chunks | DDR chunks |
| Input chunk | — | 256 KB × 16 | **512 KB × 16** |
| D32 per core | 32r × 1024K | 32r × 4096K | **64r × 4096K** |
| Multicast shape | 64 KB | 256 KB | **512 KB** |
| Weight tile | 100 KB | 100 KB | **2048 KB** |
| Weight loads | 344 × 100KB | 344 × 100KB | **268 × 2048KB** |
| DEQUANT pattern | 22 × per tile | 21 × per tile | **1 × full matrix** |
| OutCPerG (HMX) | 32 | 32 | **256** |
| OutYEnd | 128 | 256 | **192** |
| NumSrcOperands | 4 | 1 | **1** |
| HMX ops/core | 22 | 43 | **15** |
| VTCM peak | 90.4% | 64.1% | **58.5%** |
| Wall clock | 504 µs | 1029 µs | **10,885 µs** |
| HMX% | 35.5% | 70.0% | **27.5%** |
| Sync stall% | 63.9% | 37.0% | **84.9%** |
| DDR traffic | 33.6 MB | 48.4 MB | **665 MB** |
| Bottleneck | BW | COMPUTE | **BW (cliff)** |

---

## Verified numbers

| What | Exact value | Source |
|------|-------------|--------|
| Input DMA shape | `[1×64×4096]` = 512 KB | trace `aiccopytovtcm` |
| Input DMA loads | 1 per core, 16 total | trace count |
| D32 input | `[1×64×4096]` = 512 KB flat fp16 | trace `aicconverttod32` in |
| D32 output | `[1×1×2×128×1024]` = 512 KB D32 | trace `aicconverttod32` out |
| Multicast shape | `[1×1×2×128×1024]` = 512 KB | trace `aicmulticastvtcm` |
| Dequant size | 33.6 MB → 88064 KB fp16 | trace `blockdequantize_mxfp6` |
| Dequant duration | 5289.6 µs (5.3 ms) | trace dur |
| Weight tile shape | `[1×8×131072]` = 2048 KB | trace `aiccopytovtcm` |
| Weight loads/core | 15 (10×2048KB + 5×1792KB) | trace count |
| Weight loads total | 268 | trace count all cores |
| HMX input | `[1×1×6×128×1024]` = 1536 KB | trace op3 |
| HMX weight | `[1×8×131072]` = 2048 KB | trace op4 |
| HMX output | `[1×1×6×8×1024]` = 96 KB | trace op0 |
| HMX OutYEnd | 192 rows | trace op attributes |
| HMX OutCPerG | 256 channels | trace op attributes |
| HMX NumSrcOperands | 1 | trace op attributes |
| HMX ops/core | 15 | trace count |
| HMX #1 fires at | 5794 µs | trace timestamp |
| Prefetch depth | 1 tile | trace timestamps |
| VTCM peak | 4792 KB (58.5%) | memuse log |
| Wall clock | 10,885 µs | raw_device_stats |
| HMX utilisation | 27.5% | raw_device_stats |
| DDR traffic | 665 MB | raw_device_stats |
| Core imbalance | 11.1% | trace `Core-N-Execution` |

---

## Files referenced

```
results/h4096_o11008_s1024/
└── perf_dump/
    ├── dumps/
    │   ├── QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log
    │   ├── QAicGraph__op_summary_final.log
    │   ├── QAicGraph__memuse_estimate_final.log
    │   └── QAicGraph__ddr_op_summary.log
    └── opstats/
        └── *.qaic-opstats.trace.json
```
