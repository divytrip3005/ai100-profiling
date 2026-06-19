# AI100 Performance Profiling

Microbenchmark tools for profiling MatMul, MLP, and GQA operations
on Qualcomm Cloud AI 100 hardware.

---

## Scripts

### `matmul_microbenchmark.py`
Benchmarks a single MatMul `[seq × hidden] × [hidden × out]` on AI100.

**What it does:**
- Exports PyTorch model to ONNX
- Compiles to QPC binary via `qaic-compile`
- Runs profiling via `qaic-runner` + `qaic-opstats`
- Reports: ExecTime, HMX%, DDR BW, VTCM%, core imbalance, bottleneck verdict

**Usage:**
```bash
export QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-hoist-vtcm-loads=false \
  -aic-op-stats-verbosity 2 -aic-userdma-async=0 \
  -aic-hmx-async=0 -debug-glow"

python matmul_microbenchmark.py \
  --hidden-size 4096 --out-size 11008 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/h4096_o11008_s512 \
  --run-compile --dump-io --run-perf
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | 4096 | K dimension |
| `--out-size` | 11008 | N dimension |
| `--seq-len` | 1 | M dimension (1=decode, >1=prefill) |
| `--compile-num-cores` | 16 | AI100 cores |
| `--device-group` | `[0]` | Devices e.g. `[0,1,2,3]` for 4-device |
| `--no-mxfp6` | off | Disable mxfp6 weight compression |
| `--run-compile` | — | Compile ONNX → QPC |
| `--dump-io` | — | Write IO files for profiler |
| `--run-perf` | — | Run full profiling pipeline |
| `--run-hw` | — | Measure hardware latency |

**Output files (in `artifact-dir/`):**

| File | What it contains |
|------|-----------------|
| `perf_dump/opstats/*.trace.json` | Per-core execution timeline — DMA, HVX, HMX events |
| `perf_dump/dumps/QAicGraph__op_summary_final.log` | HMX tile counts and DMA loads per core |
| `perf_dump/dumps/QAicGraph__memuse_estimate_final.log` | VTCM peak usage and DDR allocation decisions |
| `perf_dump/dumps/QAicGraph__ddr_op_summary.log` | DDR traffic breakdown (weight, activation, output) |
| `perf_dump/dumps/QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log` | Compiler tiling: OutCPerG, OutYEnd, NumSrcOperands |
| `perf_dump/raw_device_stats/*.bin` | Raw PMU counters decoded by qaic-opstats |

---

### `mlp_microbenchmark.py`
Benchmarks a Llama-style SwiGLU MLP on AI100.

**What it does:**
- Benchmarks the full MLP block: gate proj + up proj + SwiGLU activation + down proj
- Same profiling pipeline as matmul_microbenchmark.py

**Usage:**
```bash
python mlp_microbenchmark.py \
  --hidden-size 4096 --intermediate-size 11008 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/mlp_h4096_s512 \
  --run-compile --dump-io --run-perf
```

---

### `gqa_microbenchmark.py`
Benchmarks Grouped-Query Attention (GQA) on AI100.

**What it does:**
- Benchmarks Q, K, V projections + attention computation
- Supports multi-head and grouped-query configurations
- Same profiling pipeline

**Usage:**
```bash
python gqa_microbenchmark.py \
  --num-heads 32 --num-kv-heads 8 --head-dim 128 --seq-len 512 \
  --compile-num-cores 16 --device-group "[0]" \
  --artifact-dir ./results/gqa_s512 \
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

## Hardware

Qualcomm Cloud AI 100 — 16 NSP cores, 8 MB VTCM/core, ~130 GB/s DDR BW, mxfp6 weights
