# MatMul Execution Walkthrough — Qualcomm AI100

**Shape:** `[512 × 4096] × [4096 × 11008]`  
**Config:** 16 cores, mxfp6=True, 1 device  
**Source:** `results/h4096_o11008_s512/perf_dump/opstats/*.trace.json`

---

## Matrices

| Tensor | Shape | Format | Size |
|--------|-------|--------|------|
| Input  | `[512 × 4096]` | fp16 | 4 MB |
| Weight | `[4096 × 11008]` | mxfp6 (compressed) | 33.6 MB |
| Output | `[512 × 11008]` | fp16 | 11 MB |

**FLOPs:** 2 × 512 × 4096 × 11008 = **46.2 billion**  
The `2×` is because each output element = K multiplications + (K−1) additions ≈ 2K operations.

All three tensors start in DDR. Nothing is in VTCM yet.

---

## Core assignment

**Strategy: N-split** (M/cores = 512/16 = 32 ≠ 64, so N-split applies)

The 11008 output channels are divided across 16 cores:

```
11008 / 16 = 688 output channels per core
```

HMX hardware processes 32 output channels per operation (hardware-fixed):

```
688 / 32 = 21.5 → 43 HMX ops per core (with slight imbalance on core 0)
43 × 16 = 688 total HMX ops
```

Source: `op_summary_final.log`
```
AICConvolutionD32 per core: [32, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]
```

---

## Step 1 — Input arrives (once per core)

**Source:** `aiccopytovtcm` event, `ts=5µs`

DMA copies one input chunk from DDR → VTCM:

```
Shape : [1 × 32 × 4096]
Size  : 32 × 4096 × 2 bytes = 256 KB
```

Only **32 rows** arrive via DMA. But HMX needs 256 rows (OutYEnd=256).
The remaining rows are assembled via multicast (Step 1b).

**Verified:** 1 input DMA load per core, 16 total across all cores (16/16 = 1 per core).

---

## Step 1b — Input distributed via multicast

**Source:** `aicmulticastvtcm` event, `ts=71µs`

The 32-row chunk is broadcast from the loading core to all other cores via the
on-chip VTCM bus. This repeats 8 times (8 chunks × 32 rows = 256 rows) to
assemble the full input tile in VTCM on every core.

```
Multicast shape : [1 × 1 × 1 × 128 × 1024] = 256 KB
After 8 rounds  : 256 rows × 4096 columns assembled in VTCM per core
```

**Verified:** 22 multicast ops per core (1 for input setup + 21 for weight distribution).

---

## Step 2 — First weight tile arrives

**Source:** `aiccopytovtcm` event, `ts=67µs`

DMA copies one weight tile from DDR → VTCM:

```
Shape (mxfp6) : [1 × 1 × 102400] = 100 KB
Covers        : 4096 K × 32 output channels (compressed)
```

Each core loads a **different** slice of the weight matrix — core 0 covers
output channels 0–31, core 1 covers 32–63, and so on.

**Verified:** 21–22 weight DMA loads per core, 344 total (344 = 11008/32 = N/OutC_tile).

---

## Step 3 — HVX decompresses weight

**Source:** `blockdequantize_mxfp6` event, `ts=105µs`

HVX expands the mxfp6 tile from 100 KB → 256 KB fp16 in VTCM:

```
Input  : [1 × 1 × 102400]  100 KB  mxfp6
Output : [1 × 1 × 131072]  256 KB  fp16
```

Now both operands for HMX are ready in VTCM:
- Input tile:  2048 KB (256 rows × 4096 K × 2 bytes)
- Weight tile:  256 KB (4096 K × 32 outC × 2 bytes)

---

## Step 4 — HMX computes one tile

**Source:** `aicconvolutiond32` event, `ts=147µs`, `dur=14.4µs`

HMX multiplies input × weight and accumulates the result:

```
Input  operand : [256 rows × 4096 K]  = 2048 KB  (in VTCM)
Weight operand : [4096 K   × 32 outC] =  256 KB  (in VTCM)
Output         : [256 rows × 32 outC] =  128 KB  (in VTCM)

OutYEnd        = 256   (rows per op)
OutCPerG       = 32    (output channels per op — hardware fixed)
InCPerG        = 4096  (full K, no splitting)
NumSrcOperands = 1     (full K in one weight operand)
```

**FLOPs per HMX op:** 2 × 256 × 4096 × 32 = **67 million**  
**Duration:** 14.4 µs (first op — later ops ~20 µs as pipeline stabilises)

---

## Step 5 — Pipelined loop over all 43 weight tiles

Steps 2–4 repeat 43 times. They are **pipelined** — while HMX computes tile N,
DMA is already loading tile N+1 (double-buffering). Observed from the timeline:

```
ts=  5µs : DMA  INPUT  [1×32×4096]   256KB
ts= 67µs : DMA  WEIGHT tile #1       100KB
ts= 71µs : MULTICAST input
ts= 85µs : DMA  WEIGHT tile #2       100KB   ← prefetch
ts=104µs : DMA  WEIGHT tile #3       100KB   ← prefetch
ts=105µs : DEQUANT tile #1  (100KB → 256KB)
ts=124µs : DMA  WEIGHT tile #4       100KB   ← prefetch
ts=147µs : HMX #1  fires  dur=14µs
ts=161µs : HMX #2  fires  dur=21µs
ts=164µs : DMA  WEIGHT tile #5       100KB
...
ts=881µs : HMX #43 fires  (last op)
```

After 43 HMX ops, core 0 has computed output for 256 rows × 688 output channels.

---

## Step 6 — Output written to DDR

**Source:** `aicconvertfromd32` × 6 events + `aiccopyfromvtcm2d` × 1 event

As HMX finishes each output tile [256 × 32] = 128 KB, HVX converts it from
D32 internal format back to fp16, then DMA writes it to DDR.

```
Output size per core : 256 rows × 688 channels × 2 bytes = 344 KB
Output location      : DDR  (11 MB total — too large for VTCM)
```

All 16 cores write simultaneously. Full `[512 × 11008]` output assembles in DDR.

---

## VTCM layout at peak

```
┌──────────────────────────────────────────────────┐
│  VTCM  (8192 KB capacity per core)               │
│                                                  │
│  Input tile     [256 × 4096]  fp16   2048 KB     │
│  Weight tile    [4096 × 32]   fp16    256 KB     │
│  Output tile    [256 × 32]    fp16    128 KB     │
│  HMX scratch    (fixed)                68 KB     │
│  ─────────────────────────────────────────────   │
│  Formula total                        2500 KB    │
│                                                  │
│  Compiler overhead (double-buffering,            │
│  D32 format buffers, multicast staging) 2753 KB  │
│  ─────────────────────────────────────────────   │
│  Actual peak measured                 5253 KB    │
│  Utilisation                          64.1%      │
│  Headroom                             2939 KB    │
└──────────────────────────────────────────────────┘
```

Source: `QAicGraph__memuse_estimate_final.log` — VTCM highwatermark lines.

---

## Time breakdown (wall clock = 1029 µs)

| Operation | % | Total µs (all cores) | What it is |
|-----------|---|----------------------|------------|
| sync_stall | 37.0% | 11,222 | HVX waiting for HMX to finish each tile |
| actual_compute | 36.5% | 11,079 | HMX multiply-accumulate |
| data_movement | 13.9% | 4,232 | DMA loading weight + input from DDR |
| weight_dequant | 9.6% | 2,918 | HVX mxfp6 → fp16 decompression |
| format_convert | 2.6% | 792 | D32 layout reformatting |

The µs column is summed across all 16 cores × all threads. DMA + HVX + HMX
**overlap** via double-buffering — that is why wall clock is only 1029 µs.

Source: `opstats/*.trace.json` event durations.

---

## Bottleneck

**COMPUTE-BOUND** (HMX% = 70%)

```
Arithmetic intensity = FLOPs / DDR bytes
                     = 46.2B / 48.4MB
                     = 916 FLOPs/byte

DDR bandwidth observed  = 46 GB/s
BW-bound minimum time   = 48.4MB / 46GB/s = 1053 µs
Actual wall clock       = 1029 µs
HMX active              = 70%
```

Shape sits at the **bandwidth ↔ compute crossover**. HMX is genuinely busy
70% of the time — not waiting for data.

---

## Verified numbers

| What | Value | Source |
|------|-------|--------|
| Strategy | N-split | `splitPlan_IntraCoreSize_final.log` |
| Output channels per core | 688 | 11008 / 16 |
| HMX ops per core | 43 | `op_summary_final.log` |
| HMX ops total | 688 | 43 × 16 |
| Each HMX op shape | `[256×4096] × [4096×32]` | trace `aicconvolutiond32` attributes |
| NumSrcOperands | 1 | trace op attributes |
| Input DMA loads per core | 1 × 256 KB `[1×32×4096]` | trace `aiccopytovtcm` |
| Weight DMA loads total | 344 × 100 KB | trace `aiccopytovtcm` |
| Output location | DDR | `memuse_estimate_final.log` |
| VTCM peak | 5253 KB (64.1%) | `memuse_estimate_final.log` |
| Wall clock | 1029 µs | `raw_device_stats` |
| HMX utilisation | 70% | `raw_device_stats` |
| Core imbalance | 9.5% | trace `Core-N-Execution` durations |

---

## Files referenced

```
results/h4096_o11008_s512/
└── perf_dump/
    ├── dumps/
    │   ├── QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log  ← core assignment
    │   ├── QAicGraph__op_summary_final.log                         ← op counts per core
    │   ├── QAicGraph__memuse_estimate_final.log                    ← VTCM usage
    │   └── QAicGraph__ddr_op_summary.log                          ← DDR traffic
    └── opstats/
        └── *.qaic-opstats.trace.json                              ← full timeline
```
