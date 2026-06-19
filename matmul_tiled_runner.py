#!/usr/bin/env python3
"""
matmul_tiled_runner.py

Decomposes a large MatMul [M × K] × [K × N] into tiles,
compiles QPCs for each unique N-slice, runs all tiles
sequentially on 1 device, and assembles the final output.

Usage:
  # Decompose [1024×4096]×[4096×22016] into [256×4096]×[4096×11008] tiles
  python matmul_tiled_runner.py \
    --M 1024 --K 4096 --N 22016 \
    --M-tile 256 --N-tile 11008

  # Also run the original large matmul for comparison
  python matmul_tiled_runner.py \
    --M 1024 --K 4096 --N 22016 \
    --M-tile 256 --N-tile 11008 \
    --compare
"""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── optional QAIC runtime ────────────────────────────────────────────────────
try:
    import qaicrt
    _qaicrt_ok = True
except ImportError:
    try:
        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt
        _qaicrt_ok = True
    except ImportError:
        _qaicrt_ok = False

try:
    import QAicApi_pb2 as aicapi
    _aicapi_ok = True
except ImportError:
    try:
        sys.path.append("/opt/qti-aic/dev/python")
        import QAicApi_pb2 as aicapi
        _aicapi_ok = True
    except ImportError:
        _aicapi_ok = False

COMPILER   = "/opt/qti-aic/exec/qaic-compile"
ONNX_OPSET = 17
WORK_DIR   = Path(__file__).parent / "tiled_runner_work"


# ── MatMul model ──────────────────────────────────────────────────────────────
class MatMulSlice(nn.Module):
    """y = x @ W  where W is a slice of the full weight matrix."""
    def __init__(self, K: int, N_slice: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(K, N_slice, dtype=torch.float16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


# ── QAIC session ──────────────────────────────────────────────────────────────
class QAICSession:
    """Thin wrapper around qaicrt for running a compiled QPC."""

    def __init__(self, qpc_path: Path):
        if not (_qaicrt_ok and _aicapi_ok):
            raise ImportError("qaicrt / QAicApi_pb2 not available.")
        dtype_map = {
            aicapi.FLOAT_TYPE:    np.float32,
            aicapi.FLOAT_16_TYPE: np.float16,
            aicapi.INT8_Q_TYPE:   np.int8,
            aicapi.INT32_I_TYPE:  np.int32,
        }
        ctx   = qaicrt.Context()
        queue = qaicrt.Queue(ctx, 0)
        qpc   = qaicrt.Qpc(str(qpc_path))
        status, iodesc_data = qpc.getIoDescriptor()
        iodesc = aicapi.IoDesc()
        iodesc.ParseFromString(bytes(iodesc_data))
        bindings = iodesc.selected_set.bindings
        idx  = {b.name: b.index for b in bindings}
        prog_props = qaicrt.QAicProgramProperties()
        try:
            prog_props.SubmitRetryTimeoutMs = 60_000
        except AttributeError:
            pass
        prog = qaicrt.Program(ctx, None, qpc, prog_props)
        prog.load(); prog.activate()
        execobj  = qaicrt.ExecObj(ctx, prog)
        qbuffers = [qaicrt.QBuffer(bytes(b.size)) for b in bindings]
        buf_dims = qaicrt.BufferDimensionsVecRef(
            [(dtype_map.get(b.type, np.float16)(0).itemsize, list(b.dims))
             for b in bindings]
        )
        self._queue    = queue
        self._execobj  = execobj
        self._qbuffers = qbuffers
        self._buf_dims = buf_dims
        self._idx      = idx
        self._bindings = bindings
        self._dtype_map = dtype_map
        self._outputs  = [b.name for b in bindings
                          if b.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    def run(self, feeds: dict) -> dict:
        for name, arr in feeds.items():
            if name not in self._idx:
                continue
            i = self._idx[name]
            self._qbuffers[i] = qaicrt.QBuffer(arr.tobytes())
            self._buf_dims[i] = (arr.itemsize,
                                 arr.shape if arr.ndim > 0 else (1,))
        if self._execobj.setData(self._qbuffers, self._buf_dims) != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("setData failed")
        if self._queue.enqueue(self._execobj) != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("enqueue failed")
        if self._execobj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("waitForCompletion failed")
        status, out_bufs = self._execobj.getData()
        return {
            name: np.frombuffer(
                bytes(out_bufs[self._idx[name]]),
                self._dtype_map.get(
                    self._bindings[self._idx[name]].type, np.float16)
            ).reshape(self._buf_dims[self._idx[name]][1])
            for name in self._outputs
        }


# ── Build + compile one QPC ───────────────────────────────────────────────────
def build_qpc(M_b: int, K: int, N_b: int, qpc_dir: Path,
              cores: int, mxfp6: bool) -> Path:
    """Export ONNX and compile to QPC. Returns path to qpc dir."""
    qpc_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = qpc_dir / f"matmul_M{M_b}_K{K}_N{N_b}.onnx"

    # build ONNX with random weights (for timing; swap real weights at runtime)
    model = MatMulSlice(K, N_b).eval()
    x     = torch.randn(1, M_b, K, dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), str(onnx_path),
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            input_names=["x"],
            output_names=["output"],
            # static shapes — no dynamic axes
        )
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        wp = onnx_path.with_suffix(".onnxweights.data")
        if wp.exists():
            wp.unlink()
        onnx.save_model(m, str(onnx_path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=wp.name,
                        size_threshold=0)
    except ImportError:
        pass

    spec_path = qpc_dir / "specializations.json"
    spec_path.write_text(json.dumps(
        {"specializations": [{"batch_size": "1", "seq_len": str(M_b)}]},
        indent=2,
    ))

    bin_dir = qpc_dir / "qpc"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)

    cmd = [
        COMPILER, "-aic-hw",
        f"-aic-hw-version=ai100",
        f"-m={onnx_path}",
        "-convert-to-fp16",
        f"-aic-num-cores={cores}",
        "-compile-only",
        f"-aic-binary-dir={bin_dir}",
    ]
    if mxfp6:
        cmd.append("-mxfp6-matmul")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Compile failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")

    return bin_dir


# ── Tiled matmul runner ───────────────────────────────────────────────────────
def run_tiled(M: int, K: int, N: int, M_b: int, N_b: int,
              input_data: np.ndarray, cores: int, mxfp6: bool,
              warmup: int = 2) -> dict:
    """
    Decompose [M×K]×[K×N] into [M_b×K]×[K×N_b] tiles.
    Compile QPCs for each unique N-slice, run all tiles, assemble output.
    """
    tiles_M = math.ceil(M  / M_b)
    tiles_N = math.ceil(N  / N_b)
    total   = tiles_M * tiles_N

    print(f"\n  Decomposition: [{M}×{K}]×[{K}×{N}]")
    print(f"  Tile size:     [{M_b}×{K}]×[{K}×{N_b}]")
    print(f"  Grid:          {tiles_M}×{tiles_N} = {total} tiles")

    # Step 1: compile one QPC per unique N-slice
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    qpc_paths = {}
    sessions  = {}

    print(f"\n  Compiling {tiles_N} QPC(s)...")
    for j in range(tiles_N):
        n_start  = j * N_b
        n_end    = min(n_start + N_b, N)
        n_actual = n_end - n_start
        key      = (M_b, K, n_actual)

        if key not in qpc_paths:
            qpc_dir  = WORK_DIR / f"qpc_M{M_b}_K{K}_N{n_actual}"
            print(f"    QPC [{M_b}×{K}]×[{K}×{n_actual}]...", end=" ", flush=True)
            qpc_path = build_qpc(M_b, K, n_actual, qpc_dir, cores, mxfp6)
            qpc_paths[key] = qpc_path
            print("done")

    # Step 2: load sessions
    if not (_qaicrt_ok and _aicapi_ok):
        print("\n  ⚠  No QAIC runtime — timing with simulated inference")
        return _simulate_tiled(M, K, N, M_b, N_b, tiles_M, tiles_N, total)

    print(f"\n  Loading sessions...")
    for key, qpc_path in qpc_paths.items():
        sessions[key] = QAICSession(qpc_path)
        print(f"    Session [{key[0]}×{key[1]}]×[{key[1]}×{key[2]}] loaded")

    # Step 3: warmup
    print(f"\n  Warming up ({warmup} passes)...")
    dummy = np.random.randn(1, M_b, K).astype(np.float16)
    for _ in range(warmup):
        for j in range(tiles_N):
            n_actual = min(N_b, N - j*N_b)
            key      = (M_b, K, n_actual)
            sessions[key].run({"x": dummy})

    # Step 4: timed run — all tiles
    output = np.zeros((M, N), dtype=np.float16)
    tile_times = []

    print(f"\n  Running {total} tiles...")
    t_total_start = time.perf_counter()

    for i in range(tiles_M):
        m_start  = i * M_b
        m_end    = min(m_start + M_b, M)
        m_actual = m_end - m_start

        # input slice for this M-tile
        x_slice = input_data[m_start:m_end, :]          # [m_actual × K]
        x_feed  = x_slice[np.newaxis, :, :]              # [1 × m_actual × K]

        for j in range(tiles_N):
            n_start  = j * N_b
            n_end    = min(n_start + N_b, N)
            n_actual = n_end - n_start
            key      = (M_b, K, n_actual)

            t0  = time.perf_counter()
            out = sessions[key].run({"x": x_feed})
            t1  = time.perf_counter()
            tile_us = (t1 - t0) * 1e6
            tile_times.append(tile_us)

            # place output tile into final output
            out_arr = list(out.values())[0]  # [1 × m_actual × n_actual]
            output[m_start:m_end, n_start:n_end] = out_arr[0]

            print(f"    T({i},{j}): rows {m_start}-{m_end-1},"
                  f" cols {n_start}-{n_end-1}"
                  f" → {tile_us:.0f}µs")

    t_total = (time.perf_counter() - t_total_start) * 1e6

    flops_total = 2 * M * K * N
    tflops      = flops_total / t_total / 1e6

    return {
        "approach":      "tiled",
        "M": M, "K": K, "N": N,
        "M_b": M_b, "N_b": N_b,
        "total_tiles":   total,
        "tile_times_us": tile_times,
        "avg_tile_us":   sum(tile_times) / len(tile_times),
        "total_us":      t_total,
        "tflops":        tflops,
        "output_shape":  list(output.shape),
        "output":        output,
    }


def _simulate_tiled(M, K, N, M_b, N_b, tiles_M, tiles_N, total):
    """Estimate timing without hardware using profiled tile time."""
    tile_log = Path("tiler_results") / f"tile_K{K}_M{M_b}_N{N_b}.log"
    tile_us  = 475.0  # fallback
    if tile_log.exists():
        try:
            from pathlib import Path as P
            txt = tile_log.read_text()
            depth=0
            for i in range(len(txt)-1,-1,-1):
                if txt[i]=='}':
                    if depth==0: end=i
                    depth+=1
                elif txt[i]=='{':
                    depth-=1
                    if depth==0:
                        rc=json.loads(txt[i:end+1]).get("report_card",{})
                        tile_us=rc.get("exec_time_us",tile_us)
                        break
        except Exception:
            pass

    total_us = total * tile_us
    flops    = 2 * M * K * N
    print(f"\n  ⚠  Simulated (no hardware) — using profiled tile time {tile_us:.0f}µs")
    print(f"  Total: {total} tiles × {tile_us:.0f}µs = {total_us:.0f}µs = {total_us/1000:.1f}ms")
    return {
        "approach": "tiled_simulated",
        "M": M, "K": K, "N": N,
        "M_b": M_b, "N_b": N_b,
        "total_tiles": total,
        "avg_tile_us": tile_us,
        "total_us": total_us,
        "tflops": flops / total_us / 1e6,
        "output_shape": [M, N],
        "output": None,
    }


# ── Original large matmul runner ──────────────────────────────────────────────
def run_original(M: int, K: int, N: int,
                 input_data: np.ndarray, cores: int, mxfp6: bool,
                 warmup: int = 2) -> dict:
    """Compile and run [M×K]×[K×N] as single large matmul."""

    print(f"\n  Compiling original [{M}×{K}]×[{K}×{N}]...", end=" ", flush=True)
    qpc_dir  = WORK_DIR / f"qpc_orig_M{M}_K{K}_N{N}"
    qpc_path = build_qpc(M, K, N, qpc_dir, cores, mxfp6)
    print("done")

    if not (_qaicrt_ok and _aicapi_ok):
        print("  ⚠  No QAIC runtime — cannot time original")
        return {"approach": "original_no_hw", "total_us": None}

    session = QAICSession(qpc_path)
    x_feed  = input_data[np.newaxis, :, :]  # [1 × M × K]

    print(f"  Warming up ({warmup} passes)...")
    for _ in range(warmup):
        session.run({"x": x_feed})

    print(f"  Running original...")
    t0  = time.perf_counter()
    out = session.run({"x": x_feed})
    t1  = time.perf_counter()
    total_us = (t1 - t0) * 1e6

    flops  = 2 * M * K * N
    tflops = flops / total_us / 1e6
    out_arr = list(out.values())[0][0]  # [M × N]

    print(f"  Original: {total_us:.0f}µs  ({tflops:.1f} TFLOPS)")
    return {
        "approach":     "original",
        "M": M, "K": K, "N": N,
        "total_us":     total_us,
        "tflops":       tflops,
        "output_shape": list(out_arr.shape),
        "output":       out_arr,
    }


# ── Print comparison ──────────────────────────────────────────────────────────
def print_comparison(orig: dict, tiled: dict):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  RESULTS: [{tiled['M']}×{tiled['K']}]×[{tiled['K']}×{tiled['N']}]")
    print(SEP)

    print(f"\n  {'Approach':<30} {'Time (µs)':>10} {'TFLOPS':>8}")
    print("  " + "-" * 52)

    orig_t = orig.get("total_us")
    tiled_t = tiled.get("total_us", 0)

    if orig_t:
        print(f"  {'Original (single call)':<30} {orig_t:>10.0f} "
              f"{orig.get('tflops',0):>8.1f}T")
    else:
        print(f"  {'Original (single call)':<30} {'N/A':>10}")

    print(f"  {'Tiled ('+str(tiled['total_tiles'])+' sequential calls)':<30} "
          f"{tiled_t:>10.0f} {tiled.get('tflops',0):>8.1f}T")

    if orig_t and tiled_t:
        speedup = orig_t / tiled_t
        print(f"\n  Speedup: {speedup:.2f}×  ({orig_t:.0f}µs → {tiled_t:.0f}µs)")
        print(f"  Time saved: {orig_t-tiled_t:.0f}µs = {(orig_t-tiled_t)/1000:.1f}ms")

    print(f"\n  Tile details:")
    print(f"    Tile shape : [{tiled['M_b']}×{tiled['K']}]×[{tiled['K']}×{tiled['N_b']}]")
    print(f"    Total tiles: {tiled['total_tiles']}")
    if "tile_times_us" in tiled:
        tts = tiled["tile_times_us"]
        print(f"    Min tile   : {min(tts):.0f}µs")
        print(f"    Max tile   : {max(tts):.0f}µs")
        print(f"    Avg tile   : {tiled['avg_tile_us']:.0f}µs")
    print(f"    Output     : {tiled['output_shape']}")

    # verify outputs match (if both available)
    if (orig.get("output") is not None and tiled.get("output") is not None):
        diff = np.abs(orig["output"].astype(np.float32) -
                      tiled["output"].astype(np.float32))
        print(f"\n  Output diff (orig vs tiled):")
        print(f"    Max abs error: {diff.max():.4f}")
        print(f"    Mean abs error: {diff.mean():.6f}")

    print(f"\n{SEP}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Run large MatMul via explicit tiling and compare with original",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--M",        type=int, required=True, help="Total rows")
    ap.add_argument("--K",        type=int, required=True, help="Hidden size")
    ap.add_argument("--N",        type=int, required=True, help="Total output cols")
    ap.add_argument("--M-tile",   type=int, required=True, help="Tile rows (M_b)")
    ap.add_argument("--N-tile",   type=int, required=True, help="Tile cols (N_b)")
    ap.add_argument("--cores",    type=int, default=16)
    ap.add_argument("--warmup",   type=int, default=2)
    ap.add_argument("--compare",  action="store_true",
                    help="Also run original large matmul for comparison")
    ap.add_argument("--no-mxfp6", action="store_true")
    args = ap.parse_args()

    mxfp6 = not args.no_mxfp6

    # create random input (same for both original and tiled)
    print(f"\nLarge matmul: [{args.M}×{args.K}]×[{args.K}×{args.N}]")
    print(f"Tile size:    [{args.M_tile}×{args.K}]×[{args.K}×{args.N_tile}]")
    print(f"\nCreating random input [{args.M}×{args.K}]...")
    input_data = np.random.randn(args.M, args.K).astype(np.float16)

    # run original
    orig = {"approach": "original_skipped", "total_us": None}
    if args.compare:
        print(f"\n{'─'*50}")
        print(f"STEP 1: Running ORIGINAL [{args.M}×{args.K}]×[{args.K}×{args.N}]")
        print(f"{'─'*50}")
        orig = run_original(args.M, args.K, args.N, input_data,
                            args.cores, mxfp6, args.warmup)

    # run tiled
    print(f"\n{'─'*50}")
    print(f"STEP 2: Running TILED ({math.ceil(args.M/args.M_tile)}×"
          f"{math.ceil(args.N/args.N_tile)} tiles)")
    print(f"{'─'*50}")
    tiled = run_tiled(args.M, args.K, args.N,
                      args.M_tile, args.N_tile,
                      input_data, args.cores, mxfp6, args.warmup)

    print_comparison(orig, tiled)


if __name__ == "__main__":
    main()
