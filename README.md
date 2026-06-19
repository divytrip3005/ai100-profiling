# AI100 Performance Profiling

Microbenchmark and profiling tools for Qualcomm Cloud AI 100 hardware.
Covers MatMul, MLP, and GQA operations.

---

## What this repo does

1. Compiles operations (MatMul / MLP / GQA) for AI100 hardware
2. Runs profiling to measure: ExecTime, HMX%, DDR BW, VTCM%, core imbalance
3. Generates a **Kernel Report Card** with bottleneck verdict and LLM implications
4. Produces detailed trace files for deeper analysis

---

## Hardware

- **Device:** Qualcomm Cloud AI 100 (AI100)
- **Cores:** 16 NSP cores per device
- **VTCM:** 8 MB per core (on-chip memory)
- **DDR BW:** ~130 GB/s
- **Precision:** fp16 compute, mxfp6 weights (default)

---

## Setup

```bash
source /opt/venv_py310/bin/activate

export QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-hoist-vtcm-loads=false \
  -aic-op-stats-verbosity 2 -aic-userdma-async=0 \
  -aic-hmx-async=0 -debug-glow"

mkdir -p /home/divytrip/tmp_dir
```

---

## Scripts

---

### `matmul_microbenchmark.py`

Benchmarks a single MatMul `[batch × seq × hidden] × [hidden × out]`.

**Usage:**
```bash
python matmul_microbenchmark.py \
  --hidden-size 4096 --out-size 11008 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/h4096_o11008_s512 \
  --run-compile --dump-io --run-perf
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | 4096 | K — input feature dimension (rows of weight) |
| `--out-size` | 11008 | N — output feature dimension (cols of weight) |
| `--seq-len` | 1 | M — sequence length (1=decode, >1=prefill) |
| `--batch-size` | 1 | Batch size |
| `--compile-num-cores` | 16 | Number of AI100 NSP cores |
| `--device-group` | `[0]` | Device IDs e.g. `[0]` or `[0,1,2,3]` |
| `--no-mxfp6` | off | Disable mxfp6 weight compression (use fp16) |
| `--run-compile` | — | Compile ONNX → QPC binary |
| `--dump-io` | — | Write input/output files needed for profiler |
| `--run-perf` | — | Run full profiling pipeline |
| `--run-hw` | — | Measure actual hardware latency (wall-clock) |
| `--hw-iters` | 50 | Iterations for latency measurement |
| `--report` | — | Run everything end-to-end, print only report card, save verbose output to `run.log` |
| `--artifact-dir` | auto | Directory to save all outputs |

---

### `mlp_microbenchmark.py`

Benchmarks a Llama-style SwiGLU MLP block (gate proj + up proj + SwiGLU + down proj).

**Usage:**
```bash
python mlp_microbenchmark.py \
  --dim 4096 --hidden-dim 11008 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/mlp_d4096_s512 \
  --run-compile --dump-io --run-perf
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dim` | 4096 | Input/output dimension (hidden size) |
| `--hidden-dim` | 11008 | Intermediate (FFN) dimension |
| `--bias` | off | Add bias to projections |
| `--fused` | off | Use fused gate+up projection |
| `--seq-len` | 1 | Sequence length |
| `--batch-size` | 1 | Batch size |
| `--compile-num-cores` | 16 | Number of AI100 NSP cores |
| `--device-group` | `[0]` | Device IDs |
| `--no-mxfp6` | off | Disable mxfp6 weight compression |
| `--run-compile` | — | Compile ONNX → QPC binary |
| `--dump-io` | — | Write input/output files for profiler |
| `--run-perf` | — | Run full profiling pipeline |
| `--run-hw` | — | Measure hardware latency |
| `--report` | — | Run everything end-to-end, print only report card, save verbose output to `run.log` |
| `--artifact-dir` | auto | Directory to save all outputs |

---

### `gqa_microbenchmark.py`

Benchmarks Grouped-Query Attention (GQA) — Q, K, V projections + scaled dot-product attention.

**Usage:**
```bash
python gqa_microbenchmark.py \
  --d-model 4096 --n-heads 32 --n-kv-heads 8 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/gqa_d4096_s512 \
  --run-compile --dump-io --run-perf
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--d-model` | 4096 | Model hidden dimension |
| `--n-heads` | 32 | Number of query heads |
| `--n-kv-heads` | 8 | Number of key/value heads (< n-heads = GQA) |
| `--causal` | off | Apply causal (autoregressive) mask |
| `--fused-attn` | off | Use fused attention kernel |
| `--seq-len` | 1 | Sequence length |
| `--batch-size` | 1 | Batch size |
| `--compile-num-cores` | 16 | Number of AI100 NSP cores |
| `--device-group` | `[0]` | Device IDs |
| `--no-mxfp6` | off | Disable mxfp6 weight compression |
| `--run-compile` | — | Compile ONNX → QPC binary |
| `--dump-io` | — | Write input/output files for profiler |
| `--run-perf` | — | Run full profiling pipeline |
| `--run-hw` | — | Measure hardware latency |
| `--report` | — | Run everything end-to-end, print only report card, save verbose output to `run.log` |
| `--artifact-dir` | auto | Directory to save all outputs |

---

## Output Files

Every run creates an `artifact-dir/` with:

```
artifact-dir/
├── *.onnx                          # Exported model
├── qpc/programqpc.bin              # Compiled hardware binary
├── specializations.json            # Shape specialization
├── io/                             # Input/output data files
└── perf_dump/
    ├── opstats/*.trace.json        # Per-core execution timeline
    ├── opstats/*.summary.txt       # Per-op cycle counts
    ├── dumps/
    │   ├── QAicGraph__op_summary_final.log          # HMX tile counts per core
    │   ├── QAicGraph__memuse_estimate_final.log      # VTCM usage per core
    │   ├── QAicGraph__ddr_op_summary.log             # DDR traffic breakdown
    │   └── QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log  # Tiling decisions
    └── raw_device_stats/*.bin      # Raw PMU counters
```

---

## Report Card Metrics

| Metric | Good | Bad |
|--------|------|-----|
| `HMX active%` | > 50% (compute-bound) | < 20% (overhead/BW-bound) |
| `VTCM residency%` | > 75% (data on-chip) | < 50% (DDR spill) |
| `DDR traffic MB` | ≈ weight size | >> weight size (redundant loads) |
| `Core imbalance%` | < 15% (balanced) | > 50% (compiler tiling issue) |
| `Bottleneck` | COMPUTE-BOUND | BANDWIDTH / OVERHEAD |

---

## Multi-device example

```bash
python matmul_microbenchmark.py \
  --hidden-size 4096 --out-size 11008 --seq-len 1024 \
  --compile-num-cores 16 --device-group "[0,1,2,3]" \
  --artifact-dir ./results/4dev_h4096_o11008_s1024 \
  --run-compile --dump-io --run-perf
```

---

## Requirements

```
Python 3.10
torch, numpy, onnx
qaicrt          → /opt/qti-aic/dev/lib/x86_64/
QAicApi_pb2     → /opt/qti-aic/dev/python/
qaic-compile    → /opt/qti-aic/exec/qaic-compile
qaic-runner     → /opt/qti-aic/exec/qaic-runner
qaic-opstats    → /opt/qti-aic/exec/qaic-opstats
```
