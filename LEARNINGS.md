# MatMul Performance Learnings — Qualcomm AI100

**Setup:** MatMul [H × 11008], mxfp6=true, batch=1, 16 cores/device  
**Shapes tested:** H=4096, H=7168 × seq_len=1,128,512,1024 × 3 configs (1dev-16core, 1dev-1core, 4dev-16core)  
**Total runs:** 24

---

## Learning 1 — Multi-device helps only at the extremes, kills you in the middle

| seq | h4096 speedup (4dev vs 1dev) | h7168 speedup |
|-----|------------------------------|---------------|
| 1   | **4.1× (103% eff)**          | **4.3× (107% eff)** |
| 128 | 1.1× (28% eff)               | 1.7× (42% eff) |
| 512 | **0.82× (SLOWER)**           | 1.25× (31% eff) |
| 1024| 3.9× (98% eff)               | 1.7× (42% eff) |

At seq=1 (decode), 4 devices gives near-perfect 4× speedup — work is purely
bandwidth-bound, 4 devices × 4 DDR buses = 4× weight loading throughput.

At seq=512 (h4096), 4 devices is **slower than 1 device** (1255µs vs 1029µs).
MDP sync overhead + 73.7% core imbalance cost more than the parallelism gains.

**Rule: Never use multi-device for prefill unless seq is very large. For decode it's worth it.**

---

## Learning 2 — h4096 seq=1024 is a compiler tiling cliff; multi-device accidentally fixes it

| Config      | ExecTime  | TFLOPS | DDR MB     | Verdict   |
|-------------|-----------|--------|------------|-----------|
| 1dev-16core | 10,885 µs | 8.5    | **665 MB** | BANDWIDTH |
| 4dev-16core | 2,776 µs  | 33.3   | 130 MB     | COMPUTE   |

On 1dev-16core, the compiler chose bad tiling for h4096×s1024 — DDR traffic
explodes to 665 MB (vs 95 MB for h7168 at same seq). The compiler was
re-loading weights from DDR repeatedly instead of keeping them in VTCM.

Switching to 4 devices forces a re-tile across 4 devices — each device handles
a smaller slice, the tile fits in VTCM, DDR traffic drops to 130 MB, and TFLOPS
jumps from 8.5 → 33.3.

**Rule: Always check DDR traffic. 665 MB for a 33 MB weight matrix means ~20×
redundant reloading. A tiling bug on one config can be masked by a different config.**

---

## Learning 3 — VTCM residency is the single best predictor of performance

| Config                    | VTCM% | HMX%  | TFLOPS | Verdict   |
|---------------------------|-------|-------|--------|-----------|
| h7168 s512  1dev-16core   | 84%   | 81.5% | 39.3   | COMPUTE   |
| h7168 s1024 1dev-16core   | 81%   | 61.3% | 26.1   | COMPUTE   |
| h4096 s1024 1dev-16core   | 46%   | 27.5% | 8.5    | BANDWIDTH ← cliff |
| h4096 s1    1dev-16core   | 31%   | 13.4% | 0.24   | BANDWIDTH |

Every time VTCM residency drops below ~50%, HMX utilization follows and
performance collapses. Above 75% VTCM → compute-bound, HMX properly utilized.

**Rule: VTCM residency is the leading indicator. Below 50% means the hardware
is spending more time fetching from DDR than computing. Fix tiling first.**

---

## Learning 4 — Single-core reveals the sync stall problem; never use it in production

On 1dev-1core, HMX utilization is surprisingly high (58–82%) but TFLOPS are
terrible (2–3 T) because sync stall grows to **65–73%** at large seq_lens.
The single core's DMA pipeline serializes all weight loads — HMX is fast but
starved waiting for data.

The paradox: 1dev-1core has *higher* HMX% than 1dev-16core at seq=1 (31% vs 13%)
but 5× worse TFLOPS. 16 cores hide memory latency via pipelining — each core
loads the next tile while the previous core computes.

**Rule: HMX% alone doesn't indicate efficiency. You need VTCM% + sync stall%
together. High HMX + high sync stall = DMA pipeline is the bottleneck, not compute.**

---

## Learning 5 — 4-device core imbalance (60–77%) is the biggest unsolved problem

Every 4dev run has 60–77% core imbalance. The slowest core takes nearly 3–4×
longer than the fastest. Straggler analysis caught 5–8 device-level outliers
and up to 20 op-level outliers per run.

Root cause: when MDP compiler splits weight matrix [H×11008] across 4 devices,
the output channel dimension (11008) doesn't divide evenly into 4×16=64 cores.
Remainder tiles get assigned unevenly.

**Rule: The 4dev efficiency loss (28–42% at mid seq_lens) is compiler-caused,
not hardware-caused. Better MDP tiling could recover 2–3× performance in the
seq=128–512 range. File a compiler issue with the imbalance data.**

---

## Learning 6 — Config selection guide by use case

| Use case                       | Best config     | Reason |
|--------------------------------|-----------------|--------|
| LLM decode (seq=1)             | **4dev-16core** | Near-perfect 4× speedup, BW-bound |
| Short prefill (seq=128)        | **1dev-16core** | 4dev only 1.1–1.7× at 28–42% eff |
| Mid prefill (seq=512, h4096)   | **1dev-16core** | 4dev is actually **slower** |
| Mid prefill (seq=512, h7168)   | **1dev-16core** | Better TFLOPS, no imbalance penalty |
| Long prefill (seq=1024, h4096) | **4dev-16core** | Compiler cliff on 1dev, 4dev 3.9× faster |
| Long prefill (seq=1024, h7168) | **1dev-16core** | 26T vs 44T, 1dev has no imbalance overhead |

---

## Learning 7 — Why seq512 performs better than seq1024: three compounding reasons

**Verified from:** `memuse_estimate_final.log`, `ddr_op_summary.log`, opstats trace JSON

### Measured data (h7168 × o11008, 1dev-16core)

| Operation | seq512 | seq1024 |
|-----------|--------|---------|
| weight_load count | 344 | **5,504** (16× more) |
| weight_load time | 5,944 µs | 16,668 µs |
| input_load time | 1,228 µs | 2,364 µs |
| output_writeback | **0** | **3,968 µs** (16 ops) |
| HMX compute time | 26,540 µs | 59,356 µs |
| ExecTimeUs | **2,058 µs** | **6,192 µs** |
| HMX% | 81.5% | 61.3% |
| VTCM% | 84.0% | 81.4% |

---

### Reason 1 — Weight tile count explodes (344 → 5504)

At seq512 the compiler fits the full K-dimension (K=7168) into VTCM — only
**344 weight tiles** needed total.

At seq1024, the input (14MB) + output (22MB) together can no longer coexist in
VTCM with the weight tile, so the compiler breaks K into smaller sub-tiles —
**5,504 weight tiles** needed, each loaded from DDR separately.

Weight load time: 5,944µs → 16,668µs (+180%).  
This is the **dominant reason** for the performance gap.

**Source:** `aiccopytovtcm` event count in trace — 344 vs 5504 weight loads.

---

### Reason 2 — Output writeback to DDR appears at seq1024

The output tensor `[1×1024×11008] = 22MB` cannot fit in VTCM (8MB limit).
The compiler writes it in **16 chunks of 1,376KB** each directly to DDR via
`aiccopyfromvtcm` — adding **3,968µs** of DDR write traffic that seq512 never pays.

seq512 output `[1×512×11008] = 11MB` — also too big for VTCM in one shot, but
the compiler finds a tiling where intermediate results stay in VTCM and the
final write is sequential. seq1024 cannot do this.

**Source:**
- `memuse_estimate_final.log`: output tensor tagged `[DDR] [no-alloc]` at seq1024,
  meaning VTCM had no room (`[no-alloc]`) so it was placed in DDR.
- `memuse_estimate_final.log`: `[shared-alloc Big overwrite]` on the 22MB output
  tensor — confirms the compiler explicitly chose DDR for this allocation.
- `aiccopyfromvtcm` events in trace: 16 events × 248µs avg = 3,968µs total.

---

### Reason 3 — HMX utilization drops (81.5% → 61.3%)

seq512 is clean — HMX runs 1,376 tiles at 19.3µs avg with minimal pipeline
interruption.

seq1024 runs 5,504 tiles at 10.8µs avg but HMX% drops because the pipeline
keeps stalling to service the 16 output writebacks (248µs each) and the larger
input chunks (147µs each vs 77µs at seq512). HMX finishes its tile but then
waits for DDR to accept the output before the next tile can start.

---

### What `[DDR] [no-alloc]` and `[shared-alloc Big overwrite]` actually mean

These tags appear in `QAicGraph__memuse_estimate_final.log` and are the compiler's
allocation decision markers — **not runtime spill indicators**:

| Tag | Meaning |
|-----|---------|
| `[VTCM]` | Compiler allocated this tensor on-chip — good |
| `[DDR] [shared-alloc Input]` | Input deliberately kept in DDR, loaded in chunks |
| `[DDR] [no-alloc]` | VTCM had no room, compiler placed this tensor in DDR |
| `[DDR] [shared-alloc Big overwrite]` | Tensor too large for VTCM, compiler chose DDR upfront |

The word **"spill"** does not appear in this compiler's logs. To detect a
tensor being placed in DDR due to size pressure, look for `[no-alloc]` combined
with `vtcmout` in the tensor name — that is the closest equivalent.

**There is no runtime spilling here — everything is planned at compile time.**

---

### Rule
> Check `memuse_estimate_final.log` for `[DDR] [no-alloc]` on output tensors.
> If the output is going to DDR, weight tile count will explode and HMX%
> will drop. The fix is to reduce seq_len per compile specialization or
> increase the number of cores so each core handles a smaller output slice
> that fits in VTCM.

---

## Quick reference — what each metric tells you

| Metric           | What it means                                      | Threshold |
|------------------|----------------------------------------------------|-----------|
| VTCM residency%  | Fraction of time data served from on-chip VTCM     | >75% = good, <50% = problem |
| HMX active%      | Fraction of time compute engines are busy          | >50% = compute-bound |
| Sync stall%      | Fraction of time waiting for DMA/inter-core sync   | >40% = pipeline problem |
| Core imbalance%  | (max_core - min_core) / max_core                   | <15% = good, >50% = compiler issue |
| DDR traffic MB   | Total data moved through DDR                       | Should be close to weight size |
| Dev outliers     | Devices crossing mean×1.5 + 1σ threshold           | 0 = balanced |
| Op outliers      | Tile ops crossing mean×1.5 + 1σ threshold          | 0 = balanced |
| `[no-alloc]` tag | Output tensor placed in DDR — weight tiles multiply | Fix: reduce seq or add cores |

---

## Files

| File                        | Description |
|-----------------------------|-------------|
| `matmul_microbenchmark.py`  | Benchmark script with straggler detection |
| `results_dashboard.html`    | Interactive filter dashboard (open in browser) |
| `results/`                  | Raw profiling data — 24 runs |
| `LEARNINGS.md`              | This file |

