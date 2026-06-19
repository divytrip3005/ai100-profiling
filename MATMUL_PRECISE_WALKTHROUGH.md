# MatMul Execution — Precise Walkthrough

**Shape:** `[512 × 4096] × [4096 × 11008]`, 16 cores, mxfp6=True  
**Source:** `results/h4096_o11008_s512/perf_dump/opstats/*.trace.json`

---

## Before anything starts

```
Input  [512 × 4096]   fp16    4 MB   → in DDR
Weight [4096 × 11008] mxfp6  33.6 MB → in DDR
Output [512 × 11008]  fp16   11 MB   → will go to DDR
```

---

## Core assignment — N-split

```
11008 output channels ÷ 16 cores = 688 channels per core
HMX processes 32 channels per call → 688/32 = 21.5
→ some cores do 21, some 22 HMX calls per pass
2 input passes × ~21.5 = 43 HMX calls per core total
43 × 16 = 688 total HMX calls
```

Source: `aicconvolutiond32` count per core from trace = 43 each.

---

## Step 1 — All 16 cores load input simultaneously

At `ts=4µs`, all 16 cores issue a DMA load at the same time:

```
Each core loads: [1 × 32 × 4096] = 32 × 4096 × 2 = 256 KB  from DDR → VTCM
```

Each core loads a **different** 32-row slice:

```
Core  0 → rows   0–31
Core  1 → rows  32–63
Core  2 → rows  64–95
...
Core 15 → rows 480–511
```

16 cores × 32 rows = **512 rows** — the entire input matrix covered in one round.

Source: 16 `aiccopytovtcm` events with shape `<1×32×4096>` all at `ts=4–5µs`.

---

## Step 2 — Input multicast: each core shares its 32 rows

At `ts=71µs`, core 0 multicasts its 32-row chunk to all other cores:

```
Multicast: [1×1×1×128×1024] = 256 KB  (one 32-row slice in D32 layout)
Source:      core 0  (1 copy)
Destinations: all 15 other cores
  opAttributes: {SrcCore: 0, DestCores: [1, 2, 3, 4, 5, 6, ...]}
  (DestCores list is truncated in the log — covers all 15 cores)
```

The trace shows 7 output operands per multicast event — these are 7 VTCM
address slots the hardware writes to simultaneously, collectively covering
all 15 destination cores through the on-chip VTCM mesh.

All 16 cores do the same simultaneously — each broadcasts its own 32 rows.
After all multicasts complete, **every core has all 512 rows** in its VTCM.

HMX `OutYEnd=256` — each HMX call processes only 256 rows at a time.
So the 512 rows are processed in **2 passes**: rows 0–255 first, then rows 256–511.

Source: 1 multicast of shape `<1×1×1×128×1024>=256KB` per core at `ts=71µs`,
`opAttributes.DestCores` confirms all 15 other cores as destinations.

---

## Step 3 — Format convert input to D32

At `ts=~71µs`: HVX converts the input from flat fp16 → D32 layout (required by HMX).  
This is `aicconverttod32` — 1 call per core.

---

## Step 4 — First weight tile loads

At `ts=67µs` (overlapping with input multicast):

```
Core 0 loads weight tile [1×1×102400] = 100 KB mxfp6  from DDR → VTCM
Covers: 4096 K × 32 output channels (channels 0–31 for core 0)
```

Each core loads a different 32-channel slice.  
Total weight tiles = N/32 = 11008/32 = **344** across all cores.  
Per core = 344/16 = **21.5 → 21 or 22 weight loads per core**.

Source: 21 `aiccopytovtcm` events per core with shape `<1×1×102400>=100KB`.

---

## Step 5 — HVX decompresses weight

HVX expands mxfp6 → fp16:

```
Input:  [1×1×102400] = 100 KB  mxfp6
Output: [1×1×131072] = 256 KB  fp16

100 KB × (16/12) ≈ 256 KB  (mxfp6: 6 bits + 8-bit scale per 16 elements)
```

Source: `blockdequantize_mxfp6` event, 21 per core.

---

## Step 6 — Weight multicast to sibling cores

After dequant, the 256 KB fp16 weight tile is multicast to sibling cores:

```
Multicast: [1×1×131072] = 256 KB fp16
```

21 such multicasts per core — one per weight tile.

Source: 21 `aicmulticastvtcm` events with shape `<1×1×131072>=256KB` per core.

**When do these multicasts happen?**

They are **not all at the beginning** — they are spread across the entire
1029 µs execution, one per weight tile, interleaved with HMX:

```
ts=147µs : HMX #1  fires
ts=151µs : MC  #1   ← fires 4µs after HMX starts consuming tile #1
ts=161µs : HMX #2  fires
ts=174µs : MC  #2
ts=183µs : HMX #3  fires
ts=203µs : MC  #3
...
ts=733µs : MC  #21  (last multicast)
ts=881µs : HMX #43  (last compute)
```

Each multicast fires just after HMX starts computing the current tile —
at that point the dequanted weight is no longer needed by this core and
can be shared with siblings while HMX still reads it. This overlaps
multicast with compute so no time is wasted.

Source: `aicmulticastvtcm` timestamps from trace, spread from ts=151µs to ts=733µs.

---

## Step 7 — HMX computes one tile

At `ts=147µs`, first HMX call fires:

```
Input  (op3): [1×1×8×128×1024] = 2048 KB  → 256 rows × 4096 K  (in VTCM)
Weight (op4): [1×1×131072]     =  256 KB  → 4096 K × 32 outC   (in VTCM)
Output (op0): [1×1×8×8×1024]   =  128 KB  → 256 rows × 32 outC  (in VTCM)

OutYEnd        = 256   rows
OutCPerG       = 32    output channels
InCPerG        = 4096  full K
NumSrcOperands = 1     (full K in one operand, no K splitting)
Duration       = 14.4 µs (first call)
```

Source: `aicconvolutiond32` op attributes from trace.

---

## Step 8 — Pipelined: DMA loads next weight while HMX computes

```
ts=  5µs : DMA  INPUT   [1×32×4096]    256KB
ts= 67µs : DMA  WEIGHT  tile #1        100KB
ts= 71µs : MULTICAST    input
ts= 85µs : DMA  WEIGHT  tile #2        100KB  ← prefetch
ts=104µs : DMA  WEIGHT  tile #3        100KB  ← prefetch
ts=105µs : DEQUANT      weight #1  (100KB → 256KB)
ts=124µs : DMA  WEIGHT  tile #4        100KB  ← prefetch
ts=147µs : HMX  #1 fires   dur=14µs
ts=161µs : HMX  #2 fires   dur=21µs
ts=164µs : DMA  WEIGHT  tile #5        100KB
...
ts=881µs : HMX  #43 fires  (last)
```

DMA is always 4–5 tiles ahead of HMX — this is **double-buffering**.

---

## Step 9 — Output written to DDR

After each HMX call, HVX converts output from D32 → fp16, then DMA writes to DDR:

```
Per HMX call:  [256 × 32]  = 128 KB output tile
Per core total: 256 rows × 688 channels × 2 bytes = 344 KB → DDR
All 16 cores:  512 rows × 11008 channels × 2 bytes = 11 MB → DDR
```

Source: 6 `aicconvertfromd32` + 1 `aiccopyfromvtcm2d` per core.

---

## VTCM at peak (measured)

```
Input tile     256 rows × 4096 K × 2 bytes   = 2048 KB
Weight tile    4096 K   × 32 outC × 2 bytes  =  256 KB
Output tile    256 rows × 32 outC × 2 bytes  =  128 KB
HMX scratch    fixed                          =   68 KB
────────────────────────────────────────────────────────
Formula total                                = 2500 KB

Compiler overhead (double-buffer, D32 format
buffers, multicast staging)                  = 2753 KB
────────────────────────────────────────────────────────
Actual peak measured                         = 5253 KB
Utilisation                                  = 64.1% of 8192 KB
```

Source: `QAicGraph__memuse_estimate_final.log` VTCM highwatermark.

---

## Verified numbers

| What | Exact value | Source |
|------|-------------|--------|
| Input DMA shape | `[1×32×4096]` = 256 KB | trace `aiccopytovtcm` |
| Input DMA loads | 1 per core, 16 total | trace count |
| Input multicast shape | `[1×1×1×128×1024]` = 256 KB | trace `aicmulticastvtcm` |
| Weight DMA shape | `[1×1×102400]` = 100 KB mxfp6 | trace `aiccopytovtcm` |
| Weight after dequant | `[1×1×131072]` = 256 KB fp16 | trace `blockdequantize_mxfp6` |
| Weight loads per core | 21 | trace count |
| Weight loads total | 344 = 11008/32 | trace count all cores |
| HMX input operand | `[1×1×8×128×1024]` = 2048 KB | trace op3 shape |
| HMX weight operand | `[1×1×131072]` = 256 KB | trace op4 shape |
| HMX output operand | `[1×1×8×8×1024]` = 128 KB | trace op0 shape |
| HMX OutYEnd | 256 rows | trace op attributes |
| HMX OutCPerG | 32 channels | trace op attributes |
| HMX InCPerG | 4096 | trace op attributes |
| HMX NumSrcOperands | 1 | trace op attributes |
| HMX ops per core | 43 | trace count |
| HMX ops total | 688 | trace count all cores |
| Input passes | 2 (256 rows each) | 512 / OutYEnd=256 |
| VTCM peak | 5253 KB (64.1%) | memuse log |
| Wall clock | 1029 µs | raw_device_stats |

---

## Files referenced

```
results/h4096_o11008_s512/
└── perf_dump/
    ├── dumps/
    │   ├── QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log
    │   ├── QAicGraph__op_summary_final.log
    │   ├── QAicGraph__memuse_estimate_final.log
    │   └── QAicGraph__ddr_op_summary.log
    └── opstats/
        └── *.qaic-opstats.trace.json
```
