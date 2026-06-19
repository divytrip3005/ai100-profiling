#!/usr/bin/env python3
"""
matmul_microbenchmark.py

Standalone microbenchmark for a single MatMul (Linear projection) kernel
targeting Qualcomm AI100 (QAIC) hardware.

Pipeline:
  1. Instantiate MatMulOnly PyTorch module and build synthetic inputs
  2. Run one CPU inference to verify correctness
  3. Export to ONNX (weight becomes a constant; only `x` is a runtime input)
  4. Emit compile artifacts: specializations.json, compile.sh
  5. [--run-compile]  run qaic-compile to produce QPC
  6. [--run-hw]       measure latency via QAICInferenceSession
  7. [--dump-io]      write .raw files + aic_batch_io.json for qaic-runner
  8. [--run-perf]     run compile_perf.sh / run_perf.sh / decode_perf.sh

Example usage:
  # Just export + emit compile script (no hardware needed)
  python matmul_microbenchmark.py --hidden-size 7168 --out-size 18432

  # Export, compile, and time on hardware
  python matmul_microbenchmark.py --hidden-size 7168 --out-size 18432 \
      --run-compile --run-hw --hw-iters 100

  # Full perf-profiling flow
  python matmul_microbenchmark.py --hidden-size 7168 --out-size 18432 \
      --run-compile --dump-io --run-perf
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# ── QAIC runtime (optional — only needed for --run-hw / --run-perf) ──────────
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

ONNX_EXPORT_OPSET = 17
COMPILER = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw"]
DEFAULT_AIC_HW_VERSION = "ai100"


# ── QAIC inference session ────────────────────────────────────────────────────
class QAICInferenceSession:
    def __init__(self, qpc_path: Path, device_ids: list[int] | None = None):
        if not (_qaicrt_ok and _aicapi_ok):
            raise ImportError("qaicrt and/or QAicApi_pb2 are unavailable.")
        self._dtype_map = {
            aicapi.FLOAT_TYPE:    np.dtype(np.float32),
            aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
            aicapi.INT8_Q_TYPE:   np.dtype(np.int8),
            aicapi.UINT8_Q_TYPE:  np.dtype(np.uint8),
            aicapi.INT16_Q_TYPE:  np.dtype(np.int16),
            aicapi.INT32_Q_TYPE:  np.dtype(np.int32),
            aicapi.INT32_I_TYPE:  np.dtype(np.int32),
            aicapi.INT64_I_TYPE:  np.dtype(np.int64),
            aicapi.INT8_TYPE:     np.dtype(np.int8),
        }
        if device_ids:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)
        qpc = qaicrt.Qpc(str(qpc_path))
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))
        self.bindings = iodesc.selected_set.bindings
        self._idx = {b.name: b.index for b in self.bindings}
        prog_props = qaicrt.QAicProgramProperties()
        try:
            prog_props.SubmitRetryTimeoutMs = 60_000
        except AttributeError:
            pass
        if device_ids and len(device_ids) > 1:
            prog_props.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, None, qpc, prog_props)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")
        self.program.activate()
        self.execObj = qaicrt.ExecObj(self.context, self.program)
        self.qbuffers = [qaicrt.QBuffer(bytes(b.size)) for b in self.bindings]
        self.buf_dims = qaicrt.BufferDimensionsVecRef(
            [(self._dtype_map[b.type].itemsize, list(b.dims)) for b in self.bindings]
        )

    @property
    def output_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    def _set_buffers(self, feeds: dict[str, np.ndarray]) -> None:
        for name, arr in feeds.items():
            if name not in self._idx:
                continue
            i = self._idx[name]
            self.qbuffers[i] = qaicrt.QBuffer(arr.tobytes())
            self.buf_dims[i] = (arr.itemsize, arr.shape if arr.ndim > 0 else (1,))

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._set_buffers(feeds)
        if self.execObj.setData(self.qbuffers, self.buf_dims) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("setData failed")
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("enqueue failed")
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError("waitForCompletion failed")
        status, out_bufs = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("getData failed")
        return {
            name: np.frombuffer(
                bytes(out_bufs[self._idx[name]]),
                self._dtype_map[self.bindings[self._idx[name]].type],
            ).reshape(self.buf_dims[self._idx[name]][1])
            for name in self.output_names
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class MatMulOnly(nn.Module):
    """y = x @ W   where W is [hidden_size, out_size]."""
    def __init__(self, hidden_size: int = 4096, out_size: int = 11008):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(hidden_size, out_size, dtype=torch.float16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


# ── Inputs ────────────────────────────────────────────────────────────────────
def build_inputs(
    hidden_size: int,
    *,
    batch_size: int = 1,
    seq_len: int = 1,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    return {"x": torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)}


# ── CPU inference (sanity check) ──────────────────────────────────────────────
def run_single_inference(
    module: MatMulOnly,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        out = module(**inputs)
    return out, {"output_shape": list(out.shape), "output_dtype": str(out.dtype)}


# ── ONNX export ───────────────────────────────────────────────────────────────
def export_onnx(
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
    onnx_path: Path,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            (inputs["x"],),
            onnx_path.as_posix(),
            opset_version=ONNX_EXPORT_OPSET,
            do_constant_folding=True,
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={
                "x":      {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            },
        )
    # Save weights as external data so the .onnx file stays small.
    try:
        import onnx
        model = onnx.load(onnx_path.as_posix())
        weights_path = onnx_path.with_suffix(".onnxweights.data")
        if weights_path.exists():
            weights_path.unlink()
        onnx.save_model(
            model,
            onnx_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=weights_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
    except ImportError:
        pass  # onnx package not installed; single-file .onnx is fine too


# ── Compile helpers ───────────────────────────────────────────────────────────
def create_specializations_json(path: Path, batch_size: int, seq_len: int) -> Path:
    path.write_text(json.dumps(
        {"specializations": [{"batch_size": str(batch_size), "seq_len": str(seq_len)}]},
        indent=2,
    ))
    return path


def create_mdp_ts_json(path: Path, device_ids: list[int], num_cores: int) -> Path | None:
    if len(device_ids) <= 1:
        return None
    payload = {
        "connections": [{"devices": list(range(len(device_ids))), "type": "p2p"}],
        "partitions": [{
            "name": "Partition0",
            "devices": [{"deviceId": i, "numCores": num_cores} for i in range(len(device_ids))],
        }],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def build_compile_command(
    onnx_path: Path,
    qpc_dir: Path,
    specialization_json: Path,
    num_cores: int,
    enable_mxfp6: bool,
    mdp_ts_json: Path | None,
) -> list[str]:
    cmd = [
        COMPILER[0], COMPILER[1],
        f"-aic-hw-version={DEFAULT_AIC_HW_VERSION}",
        f"-m={onnx_path}",
        f"-network-specialization-config={specialization_json}",
        "-convert-to-fp16",
        f"-aic-num-cores={num_cores}",
        "-compile-only",
        f"-aic-binary-dir={qpc_dir}",
    ]
    if mdp_ts_json is not None:
        cmd.append(f"-mdp-load-partition-config={mdp_ts_json}")
    if enable_mxfp6:
        cmd.append("-mxfp6-matmul")
    return cmd


def parse_device_group(s: str) -> list[int]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected format like [0] or [0,1], got: {s}")
    inner = s[1:-1].strip()
    return [int(p.strip()) for p in inner.split(",") if p.strip()] if inner else []


# ── Artifact emission ─────────────────────────────────────────────────────────
def emit_assets(
    artifact_dir: Path,
    inputs: dict[str, torch.Tensor],
    onnx_path: Path,
    compile_num_cores: int,
    batch_size: int,
    seq_len: int,
    device_ids: list[int],
    enable_mxfp6: bool,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_dir / "inputs.npz",
             x=inputs["x"].detach().cpu().numpy())

    spec_json = create_specializations_json(
        artifact_dir / "specializations.json", batch_size, seq_len)
    mdp_json = create_mdp_ts_json(
        artifact_dir / "mdp_ts_config.json", device_ids, compile_num_cores)

    qpc_dir = artifact_dir / "qpc"
    if qpc_dir.exists():
        shutil.rmtree(qpc_dir)

    cmd = build_compile_command(
        onnx_path=onnx_path,
        qpc_dir=qpc_dir,
        specialization_json=spec_json,
        num_cores=compile_num_cores,
        enable_mxfp6=enable_mxfp6,
        mdp_ts_json=mdp_json,
    )
    compile_cmd_str = " ".join(cmd)
    script = artifact_dir / "compile.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + compile_cmd_str + "\n")
    os.chmod(script, 0o755)

    return {
        "inputs_npz":          (artifact_dir / "inputs.npz").as_posix(),
        "specializations_json": spec_json.as_posix(),
        "mdp_ts_json":          mdp_json.as_posix() if mdp_json else None,
        "onnx_path":            onnx_path.as_posix(),
        "compile_command":      compile_cmd_str,
        "compile_command_list": cmd,
        "compile_script":       script.as_posix(),
    }


# ── IO dump for qaic-runner ───────────────────────────────────────────────────
def write_qaic_io(
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    io_dir: Path,
) -> Path:
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    io_list = []
    for name, tensor in inputs.items():
        arr = tensor.detach().cpu().numpy()
        arr.tofile(data_dir / f"{name}.raw")
        io_list.append({
            "path": f"data/{name}.raw",
            "io-direction": "in",
            "elem-size": arr.itemsize,
            "map-to": name,
            "dims": list(arr.shape),
        })
    out_arr = output.detach().cpu().numpy()
    out_arr.tofile(data_dir / "output.raw")
    io_list.append({
        "path": "data/output.raw",
        "io-direction": "out",
        "elem-size": out_arr.itemsize,
        "map-to": "output",
        "dims": list(out_arr.shape),
    })
    json_path = io_dir / "aic_batch_io.json"
    json_path.write_text(json.dumps({"IO-files": [io_list]}, indent=2))
    return json_path


# ── Perf-profiling scripts ────────────────────────────────────────────────────
# QAIC_COMPILER_OPTS_UNSUPPORTED flags used in every perf compile:
#   -aic-hoist-vtcm-loads=false  disable load hoisting so op placement is stable
#   -aic-op-stats-verbosity 2    emit per-op cycle stats (needed for op summary log)
#   -aic-userdma-async=0         serialise user-DMA so timing is deterministic
#   -aic-hmx-async=0             serialise HMX issue so timing is deterministic
#   -debug-glow                  dump intermediate IR / opt passes + op summary log
#   -aic-dump-graphs-dir=<dir>   write .dot files here (one per opt pass)
#   -aic-dump-dot-files          actually emit the .dot files
#
# The op summary log lands in the qpc dir as:
#   QAicGraph_*_op_summary_final.log
# That file contains total op counts, VTCM usage, cycle estimates per op.
#
# NOTE: these flags require a release-assert SDK build.
#       Always use release-assert SDKs for perf measurements.
_PERF_COMPILER_ENV_FLAGS = (
    "-aic-hoist-vtcm-loads=false"
    " -aic-op-stats-verbosity 2"
    " -aic-userdma-async=0"
    " -aic-hmx-async=0"
    " -debug-glow"
)

def write_perf_scripts(
    artifact_dir: Path,
    base_compile_cmd: list[str],
    io_json_path: Path,
    *,
    perf_num_iters: int = 20,
    perf_profile_start_iter: int = 5,
    perf_num_samples: int = 5,
    perf_stats_level: int = 70,
    perf_pmu_recipe: str = "KernelUtil",
) -> Path:
    PERF_RUNNER  = "/opt/qti-aic/exec/qaic-runner"
    PERF_OPSTATS = "/opt/qti-aic/exec/qaic-opstats"
    perf_dir    = artifact_dir / "perf_dump"
    qpc_dir     = artifact_dir / "qpc"
    stats_dir   = perf_dir / "raw_device_stats"
    opstats_dir = perf_dir / "opstats"
    dumps_dir   = perf_dir / "dumps"          # .dot files + debug-glow IR dumps
    for d in (perf_dir, stats_dir, opstats_dir, dumps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Perf compile: add profiling flags + dot-file dump flags
    perf_compile_cmd = list(base_compile_cmd) + [
        f"-stats-level={perf_stats_level}",
        "-ddr-stats",
        f"-aic-pmu-recipe={perf_pmu_recipe}",
        "-aic-perf-metrics",
    ]
    # These go via the env var, not as CLI args, because qaic-compile reads them
    # from QAIC_COMPILER_OPTS_UNSUPPORTED at startup.
    perf_compiler_env = (
        _PERF_COMPILER_ENV_FLAGS
        + f" -aic-dump-graphs-dir={dumps_dir}/"
        + " -aic-dump-dot-files"
    )

    runner_cmd = [
        PERF_RUNNER, "-t", str(qpc_dir),
        "-n", str(perf_num_iters),
        "--aic-profiling-type", "raw_device_stats",
        "--aic-profiling-start-iter", str(perf_profile_start_iter),
        "--aic-profiling-num-samples", str(perf_num_samples),
        "--aic-profiling-out-dir", str(stats_dir),
        "--aic-batch-json-input", str(io_json_path),
    ]
    opstats_cmd = [
        PERF_OPSTATS,
        "--qpc", str(qpc_dir / "programqpc.bin"),
        "--input-dir", str(stats_dir),
        "--output-dir", str(opstats_dir),
        "--summary", "--trace",
    ]
    preamble = "#!/usr/bin/env bash\nset -euo pipefail\n"
    for name, cmd in [
        ("compile_perf.sh", perf_compile_cmd),
        ("run_perf.sh",     runner_cmd),
        ("decode_perf.sh",  opstats_cmd),
    ]:
        content = preamble
        if name == "compile_perf.sh":
            content += f"rm -rf {qpc_dir}\n"
            content += f"rm -rf {dumps_dir}/*\n"
            # Set env var so qaic-compile picks up the unsupported flags.
            # Requires release-assert SDK.
            content += f"export QAIC_COMPILER_OPTS_UNSUPPORTED='{perf_compiler_env}'\n"
            content += (
                f"# Op summary log will appear in {qpc_dir}/ as:\n"
                f"#   QAicGraph_*_op_summary_final.log\n"
                f"# Dot files (one per opt pass) will appear in:\n"
                f"#   {dumps_dir}/\n"
            )
        if name == "run_perf.sh":
            content += f"rm -rf {stats_dir}/*\nrm -rf {opstats_dir}/*\n"
        content += shlex.join(str(p) for p in cmd) + "\n"
        s = perf_dir / name
        s.write_text(content)
        os.chmod(s, 0o755)
    return perf_dir


# ── Hardware run helpers ──────────────────────────────────────────────────────
def run_compile_command(compile_cmd: str) -> dict[str, Any]:
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"qaic-compile failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def ensure_compiled(artifact_dir: Path, compile_cmd: str) -> None:
    if (artifact_dir / "qpc" / "programqpc.bin").is_file():
        return
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"qaic-compile failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_hw_session(
    qpc_path: Path,
    inputs: dict[str, torch.Tensor],
    device_ids: list[int],
    warmup: int,
    iters: int,
) -> float:
    """Returns average latency in milliseconds."""
    if not (_qaicrt_ok and _aicapi_ok):
        raise RuntimeError("QAIC runtime is not available in this environment.")
    session = QAICInferenceSession(qpc_path, device_ids=device_ids or None)
    feed = {name: t.detach().cpu().numpy() for name, t in inputs.items()}
    for _ in range(warmup):
        session.run(feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(feed)
    return (time.perf_counter() - t0) * 1000.0 / iters


# ── Kernel report card ────────────────────────────────────────────────────────
_TIME_BUCKETS: dict[str, set[str]] = {
    "actual_compute": {"aicconvolutiond32"},
    "weight_dequant": {"blockdequantize_mxfp6"},
    "data_movement":  {"aiccopysamevtcm", "aicmulticastvtcm"},
    "format_convert": {"aicconverttod32", "aicconvertfromd32"},
    "sync_stall":     {"sync HMX", "sync HVX", "sync DMAIssue"},
    "semaphore":      {"aicinputsemaphoreinc", "aicoutputsemaphoreinc"},
    "profiling_oh":   {"aicendcyclestats"},
}

_BUCKET_LABELS: dict[str, str] = {
    "actual_compute": "Actual compute   (aicconvolutiond32)",
    "weight_dequant": "Weight dequant   (blockdequantize_mxfp6)",
    "data_movement":  "Data movement    (DMA copy + multicast)",
    "format_convert": "Format convert   (D32 reformat)",
    "sync_stall":     "Sync stall       (HMX/HVX/DMA wait)",
    "semaphore":      "Semaphore        (input/output signal)",
    "profiling_oh":   "Profiling OH     (aicendcyclestats -- not in prod)",
}


def parse_runner_metrics(stdout: str) -> dict[str, float]:
    """Extract avg values from qaic-runner Aggregated Device Metrics section.

    Handles both single-device (ExecTimeUs_Func_0) and multi-device
    (ExecTimeUs_Dev_N_Func_0) naming. Aggregation per metric:
      ExecTimeUs      -> max  (wall-clock = slowest device)
      DDRTrafficMB    -> sum  (each device moves its own share)
      everything else -> avg  (utilization pcts, rates)
    """
    import re
    per_device: dict[str, dict[str, float]] = {}
    in_agg = False
    for line in stdout.splitlines():
        if "Aggregated Device Metrics Report" in line:
            in_agg = True
            continue
        if not in_agg:
            continue
        if line.strip().startswith("Metric"):
            continue
        parts = [p.strip().rstrip(",") for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            val = float(parts[1])
        except ValueError:
            continue
        raw = parts[0].replace("_Func_0", "")
        m = re.match(r"^(.+?)_Dev_(\d+)$", raw)
        base, dev = (m.group(1), m.group(2)) if m else (raw, "0")
        per_device.setdefault(base, {})[dev] = val

    metrics: dict[str, float] = {}
    for base, dev_vals in per_device.items():
        vals = list(dev_vals.values())
        if "ExecTimeUs" in base:
            metrics[base] = max(vals)
        elif "DDRTrafficMB" in base:
            metrics[base] = sum(vals)
        else:
            metrics[base] = sum(vals) / len(vals)
    return metrics


def _outlier_threshold(vals: list[float]) -> float:
    """Return the outlier threshold: mean×1.5 AND mean+1×std (return the higher bar)."""
    if not vals:
        return float("inf")
    mean = sum(vals) / len(vals)
    std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return max(mean * 1.5, mean + std)


def parse_device_straggler(runner_stdout: str) -> dict[str, Any]:
    """Per-op imbalance map across devices.

    For each metric base (e.g. ExecTimeUs, HMXActivePct) that has per-device
    values, check if any device is an outlier (dur > mean×1.5 AND mean+1×std).
    Returns only metrics where at least one device is an outlier.
    """
    import re
    per_device: dict[str, dict[str, float]] = {}
    in_agg = False
    for line in runner_stdout.splitlines():
        if "Aggregated Device Metrics Report" in line:
            in_agg = True
            continue
        if not in_agg:
            continue
        if line.strip().startswith("Metric"):
            continue
        parts = [p.strip().rstrip(",") for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            val = float(parts[1])
        except ValueError:
            continue
        raw = parts[0].replace("_Func_0", "")
        m = re.match(r"^(.+?)_Dev_(\d+)$", raw)
        if not m:
            continue  # single-device run — no inter-device analysis needed
        base, dev = m.group(1), m.group(2)
        per_device.setdefault(base, {})[dev] = val

    if not per_device:
        return {"num_devices": 1, "outliers": []}

    num_devices = max(len(v) for v in per_device.values())
    outliers: list[dict[str, Any]] = []

    for base, dev_vals in per_device.items():
        vals = list(dev_vals.values())
        if len(vals) < 2:
            continue
        threshold = _outlier_threshold(vals)
        mean = sum(vals) / len(vals)
        for dev, val in dev_vals.items():
            if val > threshold:
                outliers.append({
                    "metric":    base,
                    "device":    dev,
                    "value":     round(val, 3),
                    "mean":      round(mean, 3),
                    "pct_above": round(100.0 * (val - mean) / mean, 1),
                    "severity":  "CRITICAL" if val > mean * 3.0 else "WARNING",
                })

    outliers.sort(key=lambda x: -x["pct_above"])
    return {"num_devices": num_devices, "outliers": outliers}


def parse_op_straggler(trace_paths: list[Path]) -> dict[str, Any]:
    """Per-tile op imbalance across cores (from one or more trace JSONs).

    For each aicconvolutiond32 op index, compare duration across all cores.
    Flag cores where duration > mean×1.5 AND mean+1×std.
    """
    import re
    # core_ops[core_idx] = list of (op_seq, dur_us)
    core_ops: dict[int, list[tuple[int, float]]] = {}

    for trace_path in trace_paths:
        try:
            with open(trace_path) as f:
                data = json.load(f)
        except Exception:
            continue
        events = data.get("traceEvents", [])

        tid_to_core: dict[int, int] = {}
        for e in events:
            if e.get("ph") == "M" and e.get("name") == "thread_name":
                tname = e.get("args", {}).get("name", "")
                m = re.match(r"QAicGraph.*_Core_(\d+)", tname)
                if m:
                    tid_to_core[e.get("tid")] = int(m.group(1))

        # collect aicconvolutiond32 events per core in order
        core_seq_counter: dict[int, int] = {}
        for e in events:
            if e.get("ph") != "X" or e.get("name") != "aicconvolutiond32":
                continue
            core = tid_to_core.get(e.get("tid"))
            if core is None:
                continue
            seq = core_seq_counter.get(core, 0)
            core_seq_counter[core] = seq + 1
            dur = float(e.get("dur", 0.0))
            core_ops.setdefault(core, []).append((seq, dur))

    if not core_ops:
        return {"outliers": []}

    # build op_seq -> {core: dur}
    op_by_seq: dict[int, dict[int, float]] = {}
    for core, ops in core_ops.items():
        for seq, dur in ops:
            op_by_seq.setdefault(seq, {})[core] = dur

    outliers: list[dict[str, Any]] = []
    for seq, core_durs in op_by_seq.items():
        vals = list(core_durs.values())
        if len(vals) < 2:
            continue
        threshold = _outlier_threshold(vals)
        mean = sum(vals) / len(vals)
        for core, dur in core_durs.items():
            if dur > threshold:
                outliers.append({
                    "op_seq":    seq,
                    "core":      core,
                    "dur_us":    round(dur, 3),
                    "mean_us":   round(mean, 3),
                    "pct_above": round(100.0 * (dur - mean) / mean, 1),
                    "severity":  "CRITICAL" if dur > mean * 3.0 else "WARNING",
                })

    outliers.sort(key=lambda x: -x["pct_above"])
    return {"outliers": outliers[:20]}  # cap at top-20 to avoid flooding


def parse_trace_breakdown(trace_path: Path) -> dict[str, Any]:
    """Parse a single trace JSON: time buckets, core imbalance, VTCM residency."""
    with open(trace_path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    bucket_us: dict[str, float] = {k: 0.0 for k in _TIME_BUCKETS}
    tcm_us = ddr_us = 0.0

    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        args = e.get("args", {})

        for bucket, op_set in _TIME_BUCKETS.items():
            if name in op_set:
                # sync events store wait time in opSyncDurUs; dur is 0
                if name.startswith("sync "):
                    dur = float(args.get("opSyncDurUs", 0.0))
                else:
                    dur = float(e.get("dur", 0.0))
                bucket_us[bucket] += dur
                break

        # VTCM residency: TCM vs DDR for real compute ops only
        if name not in {"aicendcyclestats"} and not name.startswith("sync ") \
                and not name.startswith("Core-"):
            mem = args.get("opMemory", "")
            dur = float(e.get("dur", 0.0))
            if mem == "TCM":
                tcm_us += dur
            elif mem == "DDR":
                ddr_us += dur

    # ── Core imbalance — adjusted to exclude aicendcyclestats overhead ────────
    #
    # Core-N-Execution duration includes the profiling flush (aicendcyclestats)
    # at the end of each core. All 16 cores race to write to DDR simultaneously,
    # causing random contention — this is NOT real computation imbalance.
    #
    # Fix: per core, real computation ends when the FIRST aicendcyclestats
    # starts. Real duration = earliest_aicendcyclestats_start - core_start.
    # Each core gets its own subtraction since DDR wait time differs per core.

    # Step 1 — build tid → core_index from thread_name metadata events
    import re
    tid_to_core: dict[int, int] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname = e.get("args", {}).get("name", "")
            m = re.match(r"QAicGraph__Core_(\d+)", tname)
            if m:
                tid_to_core[e.get("tid")] = int(m.group(1))

    # Step 2 — collect earliest aicendcyclestats start time per core
    earliest_stats_ts: dict[int, float] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("name") == "aicendcyclestats":
            core = tid_to_core.get(e.get("tid"))
            if core is not None:
                ts = float(e.get("ts", 0.0))
                if core not in earliest_stats_ts or ts < earliest_stats_ts[core]:
                    earliest_stats_ts[core] = ts

    # Step 3 — adjusted duration = core_start to earliest_stats_start
    core_real_durs: list[float] = []
    for e in events:
        if e.get("ph") == "X" \
                and e.get("name", "").startswith("Core-") \
                and e.get("name", "").endswith("-Execution"):
            m = re.match(r"Core-(\d+)-Execution", e.get("name", ""))
            if not m:
                continue
            core_idx  = int(m.group(1))
            core_start = float(e.get("ts", 0.0))
            core_dur   = float(e.get("dur", 0.0))
            if core_idx in earliest_stats_ts:
                real_dur = earliest_stats_ts[core_idx] - core_start
            else:
                real_dur = core_dur          # fallback if no stats event found
            if real_dur > 0:
                core_real_durs.append(real_dur)

    total_us = sum(bucket_us.values())
    buckets = {
        k: {
            "dur_us": round(v, 3),
            "pct":    round(100.0 * v / total_us, 1) if total_us > 0 else 0.0,
        }
        for k, v in bucket_us.items()
    }
    return {
        "buckets":            buckets,
        "total_accounted_us": round(total_us, 3),
        "core_imbalance_pct": round(100.0 * (max(core_real_durs) - min(core_real_durs)) / max(core_real_durs), 1)
                              if core_real_durs else 0.0,
        "fastest_core_us":    round(min(core_real_durs), 3) if core_real_durs else 0.0,
        "slowest_core_us":    round(max(core_real_durs), 3) if core_real_durs else 0.0,
        "vtcm_residency_pct": round(100.0 * tcm_us / (tcm_us + ddr_us), 1)
                              if (tcm_us + ddr_us) > 0 else 0.0,
    }


def build_report_card(
    args: argparse.Namespace,
    hw_results: Optional[dict[str, Any]],
    runner_metrics: dict[str, float],
    trace_breakdown: dict[str, Any],
    device_straggler: Optional[dict[str, Any]] = None,
    op_straggler: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    H, O  = args.hidden_size, args.out_size
    B, S  = args.batch_size, args.seq_len
    mxfp6 = not args.no_mxfp6
    flops = 2 * B * S * H * O

    weight_bytes = H * O * (0.75 if mxfp6 else 2.0)   # mxfp6=6bit=0.75B, fp16=2B
    input_bytes  = B * S * H * 2.0
    output_bytes = B * S * O * 2.0
    arith_intensity = flops / (weight_bytes + input_bytes + output_bytes)

    hmx_pct  = runner_metrics.get("HMXActivePct")
    sync_pct = trace_breakdown.get("buckets", {}).get("sync_stall", {}).get("pct", 0.0)

    if hmx_pct is None:
        bottleneck      = "UNKNOWN  (run with --run-perf to classify)"
        primary_limiter = "unknown"
    elif hmx_pct < 10:
        bottleneck      = "OVERHEAD-BOUND"
        primary_limiter = (
            f"sync/DMA stall ({sync_pct:.0f}% of time)" if sync_pct > 30
            else "fixed per-call overhead — shape too small for this hardware"
        )
    elif hmx_pct < 50:
        bottleneck      = "BANDWIDTH-BOUND"
        ddr_bw = runner_metrics.get("DDRBandwidthGBPerSec", 0.0)
        primary_limiter = f"DDR weight/activation movement ({ddr_bw:.2f} GB/s)"
    else:
        bottleneck      = "COMPUTE-BOUND"
        primary_limiter = f"HMX engine ({hmx_pct:.1f}% active)"

    return {
        "shape":            f"[{B}, {S}, {H}] x [{H}, {O}]",
        "mxfp6":            mxfp6,
        "flops":            flops,
        "arith_intensity":  round(arith_intensity, 3),
        "weight_bytes":     weight_bytes,
        "activation_bytes": input_bytes + output_bytes,
        "bottleneck":       bottleneck,
        "primary_limiter":  primary_limiter,
        "hmx_active_pct":   hmx_pct,
        "hvx_active_pct":   runner_metrics.get("HVXActivePct"),
        "dma_active_pct":   runner_metrics.get("DMAActivePct"),
        "ddr_bw_gbs":       runner_metrics.get("DDRBandwidthGBPerSec"),
        "ddr_traffic_mb":   runner_metrics.get("DDRTrafficMB"),
        "exec_time_us":     runner_metrics.get("ExecTimeUs"),
        "latency_ms":       hw_results.get("latency_ms")      if hw_results else None,
        "tflops_achieved":  hw_results.get("tflops_achieved") if hw_results else None,
        "time_breakdown":        trace_breakdown.get("buckets", {}),
        "core_imbalance_pct":    trace_breakdown.get("core_imbalance_pct"),
        "fastest_core_us":       trace_breakdown.get("fastest_core_us"),
        "slowest_core_us":       trace_breakdown.get("slowest_core_us"),
        "vtcm_residency_pct":    trace_breakdown.get("vtcm_residency_pct"),
        "device_straggler":      device_straggler or {"num_devices": 1, "outliers": []},
        "op_straggler":          op_straggler     or {"outliers": []},
    }


def print_report_card(rc: dict[str, Any]) -> None:
    def bar(pct: Optional[float], width: int = 20) -> str:
        if pct is None:
            return "?" * width
        filled = max(0, min(width, round(pct / 100 * width)))
        return "█" * filled + "░" * (width - filled)

    def fmt(v: Any, spec: str = ".3f") -> str:
        return format(v, spec) if v is not None else "N/A"

    W   = 68
    SEP = "=" * W
    DIV = "-" * 60

    print(f"\n{SEP}")
    print(f"  KERNEL REPORT CARD  --  MatMul  {rc['shape']}")
    print(f"  mxfp6={rc['mxfp6']}  "
          f"FLOPs={rc['flops']:,}  "
          f"Arith.Intensity={fmt(rc['arith_intensity'])} FLOPs/byte")
    print(SEP)

    print(f"\n  VERDICT : {rc['bottleneck']}")
    print(f"  Limiter : {rc['primary_limiter']}")

    print(f"\n  {DIV}")
    print(f"  PERFORMANCE")
    print(f"  {DIV}")
    print(f"  Latency (hw session)   : {fmt(rc['latency_ms'],      '.4f')} ms")
    print(f"  ExecTime (qaic-runner) : {fmt(rc['exec_time_us'],    '.3f')} us")
    print(f"  Achieved TFLOPS        : {fmt(rc['tflops_achieved'], '.4f')}")

    print(f"\n  {DIV}")
    print(f"  BOTTLENECK SIGNALS")
    print(f"  {DIV}")
    print(f"  HMX Active : {bar(rc['hmx_active_pct'])}  {fmt(rc['hmx_active_pct'], '.1f')}%"
          f"  (>50% = compute-bound, <10% = overhead-bound)")
    print(f"  HVX Active : {bar(rc['hvx_active_pct'])}  {fmt(rc['hvx_active_pct'], '.1f')}%")
    print(f"  DMA Active : {bar(rc['dma_active_pct'])}  {fmt(rc['dma_active_pct'], '.1f')}%")
    print(f"  DDR BW     : {fmt(rc['ddr_bw_gbs'], '.2f')} GB/s"
          f"   DDR Traffic: {fmt(rc['ddr_traffic_mb'], '.3f')} MB")

    print(f"\n  {DIV}")
    print(f"  TIME BREAKDOWN  (summed across all cores x threads)")
    print(f"  {DIV}")
    breakdown = rc.get("time_breakdown", {})
    for key, vals in sorted(breakdown.items(), key=lambda x: -x[1].get("pct", 0)):
        pct = vals.get("pct",    0.0)
        dur = vals.get("dur_us", 0.0)
        label = _BUCKET_LABELS.get(key, key)
        print(f"  {bar(pct, 16)}  {pct:5.1f}%  {dur:8.2f}us  {label}")

    print(f"\n  {DIV}")
    print(f"  HARDWARE FIT")
    print(f"  {DIV}")
    imb  = rc.get("core_imbalance_pct")
    fast = rc.get("fastest_core_us")
    slow = rc.get("slowest_core_us")
    vtcm = rc.get("vtcm_residency_pct")
    imb_flag  = "[POOR  -- compiler tiling uneven]" if (imb  and imb  > 15) else "[OK]"
    vtcm_flag = "[GOOD  -- on-chip]"                if (vtcm and vtcm > 70) else "[CHECK -- DDR spill]"
    print(f"  Core imbalance  : {fmt(imb,  '.1f')}%"
          f"  [{fmt(fast, '.1f')}us - {fmt(slow, '.1f')}us]  {imb_flag}")
    print(f"  VTCM residency  : {fmt(vtcm, '.1f')}%  {vtcm_flag}")

    # ── Straggler imbalance map ───────────────────────────────────────────────
    dev_strag = rc.get("device_straggler", {})
    op_strag  = rc.get("op_straggler",     {})
    dev_outliers = dev_strag.get("outliers", [])
    op_outliers  = op_strag.get("outliers",  [])
    num_devices  = dev_strag.get("num_devices", 1)

    print(f"\n  {DIV}")
    print(f"  STRAGGLER IMBALANCE MAP")
    print(f"  {DIV}")
    print(f"  Threshold: mean×1.5 AND mean+1×std  |  WARNING=>1.5x  CRITICAL=>3x")

    if num_devices > 1:
        if not dev_outliers:
            print(f"  [INTER-DEVICE]  All {num_devices} devices within threshold -- balanced")
        else:
            print(f"  [INTER-DEVICE]  {len(dev_outliers)} outlier(s) across {num_devices} devices:")
            print(f"  {'Metric':<30} {'Dev':>5}  {'Value':>10}  {'Mean':>10}  {'%Above':>8}  Severity")
            print(f"  {'-'*30} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  --------")
            for o in dev_outliers[:10]:
                sev = "*** CRITICAL" if o["severity"] == "CRITICAL" else "  ! WARNING"
                print(f"  {o['metric']:<30} {o['device']:>5}  "
                      f"{o['value']:>10.3f}  {o['mean']:>10.3f}  "
                      f"{o['pct_above']:>7.1f}%  {sev}")
    else:
        print(f"  [INTER-DEVICE]  Single-device run -- no inter-device analysis")

    if not op_outliers:
        print(f"  [INTRA-DEVICE]  All tile ops within threshold -- balanced")
    else:
        print(f"  [INTRA-DEVICE]  {len(op_outliers)} outlier tile op(s) (top 20 shown):")
        print(f"  {'TileOp#':>8}  {'Core':>6}  {'Dur(us)':>9}  {'Mean(us)':>9}  {'%Above':>8}  Severity")
        print(f"  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  --------")
        for o in op_outliers:
            sev = "*** CRITICAL" if o["severity"] == "CRITICAL" else "  ! WARNING"
            print(f"  {o['op_seq']:>8}  {o['core']:>6}  "
                  f"{o['dur_us']:>9.3f}  {o['mean_us']:>9.3f}  "
                  f"{o['pct_above']:>7.1f}%  {sev}")

    print(f"\n  {DIV}")
    print(f"  LLM IMPLICATION")
    print(f"  {DIV}")
    bottleneck = rc["bottleneck"]
    exec_us    = rc.get("exec_time_us") or 0.0
    imb_pct    = rc.get("core_imbalance_pct") or 0.0
    if "OVERHEAD" in bottleneck:
        total_calls = 61 * 6
        floor_ms    = exec_us * total_calls / 1000.0
        imb_waste   = exec_us * (imb_pct / 100) * total_calls / 1000.0
        print(f"  Shape too small -- hardware overhead dominates real compute.")
        print(f"  61 layers x 6 matmuls/layer = {total_calls} calls/token.")
        print(f"  At this shape: ~{floor_ms:.2f} ms/token latency floor.")
        print(f"  Core imbalance alone wastes ~{imb_waste:.2f} ms/token across all layers.")
        print(f"  --> Run at actual LLM shapes (e.g. 7168x18432) for real numbers.")
    elif "BANDWIDTH" in bottleneck:
        ddr = rc.get("ddr_bw_gbs") or 0.0
        print(f"  Decode is weight-movement dominated ({ddr:.2f} GB/s DDR BW).")
        print(f"  mxfp6 gives ~2.67x weight size reduction -> ~2.67x decode speedup.")
        print(f"  Core imbalance ({fmt(imb, '.1f')}%) compounds across 61 layers.")
        print(f"  --> Try: fewer cores, larger batch, fused dequant+matmul.")
    elif "COMPUTE" in bottleneck:
        print(f"  HMX is the bottleneck -- good utilization at this seq_len.")
        print(f"  This is the prefill regime; decode will be bandwidth-bound.")
        print(f"  --> Focus: tile size tuning, core count sweep, op fusion.")
    print(f"\n{SEP}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MatMul microbenchmark for Qualcomm AI100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model shape
    p.add_argument("--hidden-size",       type=int,  default=4096,
                   help="Input feature dimension (rows of W).")
    p.add_argument("--out-size",          type=int,  default=11008,
                   help="Output feature dimension (cols of W).")
    # Batch / sequence
    p.add_argument("--batch-size",        type=int,  default=1)
    p.add_argument("--seq-len",           type=int,  default=1,
                   help="Tokens per sequence (1 = decode, >1 = prefill).")
    # Compile
    p.add_argument("--compile-num-cores", type=int,  default=16,
                   help="AI100 core count passed to qaic-compile.")
    p.add_argument("--device-group",      type=str,  default="[0]",
                   help="Execution devices, e.g. [0] or [0,1].")
    p.add_argument("--no-mxfp6",         action="store_true",
                   help="Disable -mxfp6-matmul (enabled by default).")
    # Hardware timing
    p.add_argument("--hw-warmup",         type=int,  default=5,
                   help="Warmup iterations before timing.")
    p.add_argument("--hw-iters",          type=int,  default=50,
                   help="Timed iterations for latency measurement.")
    # Actions
    p.add_argument("--run-compile",       action="store_true",
                   help="Invoke qaic-compile after exporting ONNX.")
    p.add_argument("--run-hw",            action="store_true",
                   help="Run compiled QPC on hardware and report latency.")
    p.add_argument("--dump-io",           action="store_true",
                   help="Write .raw files + aic_batch_io.json for qaic-runner.")
    p.add_argument("--run-perf",          action="store_true",
                   help="Run perf profiling scripts (requires --dump-io).")
    p.add_argument(
        "--artifact-dir", type=Path,
        default=Path(__file__).with_suffix(""),
        help="Directory for ONNX, inputs, specializations, QPC, and scripts.",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Run everything end-to-end and print only the report card. "
             "Implies --run-compile --dump-io --run-perf. "
             "All verbose output is saved to artifact-dir/run.log.",
    )
    return p


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = build_parser().parse_args()

    # --report implies all execution steps and quiet mode
    if args.report:
        args.run_compile = True
        args.dump_io     = True
        args.run_perf    = True

    if args.run_perf and not args.dump_io:
        raise ValueError("--run-perf requires --dump-io.")

    # In --report mode all verbose output goes to run.log, not stdout
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.artifact_dir / "run.log" if args.report else None

    def vprint(*a, **kw) -> None:
        if log_path:
            with open(log_path, "a") as f:
                print(*a, **kw, file=f)
        else:
            print(*a, **kw)

    def vwrite(text: str) -> None:
        if log_path:
            with open(log_path, "a") as f:
                f.write(text)
        else:
            sys.stdout.write(text)

    if log_path:
        log_path.write_text("")          # clear previous run
        vprint(f"=== run started ===  shape: [{args.batch_size},{args.seq_len},{args.hidden_size}]x[{args.hidden_size},{args.out_size}]")

    # ── 1. Build module and inputs ────────────────────────────────────────────
    module = MatMulOnly(hidden_size=args.hidden_size, out_size=args.out_size)
    module.eval()

    inputs = build_inputs(
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=torch.float16,
    )

    # ── 2. CPU inference (correctness check) ──────────────────────────────────
    output, inference_meta = run_single_inference(module, inputs)
    flops = 2 * args.batch_size * args.seq_len * args.hidden_size * args.out_size
    inference_meta["flops"] = flops

    # ── 3. ONNX export ────────────────────────────────────────────────────────
    onnx_path = args.artifact_dir / (
        f"matmul_h{args.hidden_size}_o{args.out_size}"
        f"_b{args.batch_size}_s{args.seq_len}.onnx"
    )
    export_onnx(module, inputs, onnx_path)
    vprint(f"ONNX exported  : {onnx_path}")

    # ── 4. Compile artifacts ──────────────────────────────────────────────────
    device_ids = parse_device_group(args.device_group)
    assets = emit_assets(
        artifact_dir=args.artifact_dir,
        inputs=inputs,
        onnx_path=onnx_path,
        compile_num_cores=args.compile_num_cores,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device_ids=device_ids,
        enable_mxfp6=not args.no_mxfp6,
    )
    vprint(f"Compile script : {assets['compile_script']}")

    result: dict[str, Any] = {
        "model": {
            "hidden_size": args.hidden_size,
            "out_size": args.out_size,
            "weight_shape": [args.hidden_size, args.out_size],
        },
        "run_config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "mxfp6": not args.no_mxfp6,
            "compile_num_cores": args.compile_num_cores,
        },
        "single_inference": inference_meta,
        "assets": assets,
    }

    # ── 5. IO dump ────────────────────────────────────────────────────────────
    io_json_path = None
    perf_dir = None
    if args.dump_io:
        io_dir = args.artifact_dir / "io"
        io_json_path = write_qaic_io(inputs, output, io_dir)
        vprint(f"IO dumped      : {io_dir}")
        vprint(f"aic_batch_io   : {io_json_path}")
        result["io_json"] = io_json_path.as_posix()
        perf_dir = write_perf_scripts(
            args.artifact_dir, assets["compile_command_list"], io_json_path
        )
        vprint(f"Perf scripts   : {perf_dir}/{{compile_perf,run_perf,decode_perf}}.sh")
        result["perf_dir"] = perf_dir.as_posix()

    # ── 6. Compile ────────────────────────────────────────────────────────────
    if args.run_compile:
        vprint("Running qaic-compile ...")
        compile_result = run_compile_command(assets["compile_command"])
        result["compile_result"] = compile_result
        vprint("Compile done.")

    # ── 7. Hardware latency ───────────────────────────────────────────────────
    if args.run_hw:
        ensure_compiled(args.artifact_dir, assets["compile_command"])
        ms = run_hw_session(
            qpc_path=args.artifact_dir / "qpc",
            inputs=inputs,
            device_ids=device_ids,
            warmup=args.hw_warmup,
            iters=args.hw_iters,
        )
        tflops = (flops / ms / 1e9) if ms > 0 else 0.0
        result["hw_results"] = {
            "latency_ms":      round(ms, 4),
            "tflops_achieved": round(tflops, 3),
        }
        vprint(f"HW latency     : {ms:.4f} ms  ({tflops:.3f} TFLOPS)")

    # ── 8. Perf profiling ─────────────────────────────────────────────────────
    runner_metrics:  dict[str, float] = {}
    runner_stdout:   str              = ""
    trace_breakdown: dict[str, Any]   = {}

    if args.run_perf:
        if perf_dir is None:
            raise RuntimeError("perf_dir is None — --dump-io must have run first.")
        for script_name in ("compile_perf.sh", "run_perf.sh", "decode_perf.sh"):
            script = perf_dir / script_name
            vprint(f"Running: {script}")
            proc = subprocess.run(["bash", str(script)], text=True, capture_output=True)
            vwrite(proc.stdout)
            if proc.stderr:
                vwrite(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"{script_name} failed (rc={proc.returncode})")
            if script_name == "run_perf.sh":
                runner_stdout  = proc.stdout
                runner_metrics = parse_runner_metrics(proc.stdout)
        vprint(f"Perf results   : {perf_dir}/opstats/")
        trace_files = sorted((perf_dir / "opstats").glob("*.trace.json"))
        if trace_files:
            mid_trace = trace_files[len(trace_files) // 2]
            trace_breakdown = parse_trace_breakdown(mid_trace)
            result["trace_breakdown_source"] = mid_trace.name

    # ── 9. Report card ────────────────────────────────────────────────────────
    hw_results = result.get("hw_results")
    if hw_results or runner_metrics:
        device_straggler = parse_device_straggler(runner_stdout) if runner_stdout else None
        op_straggler     = parse_op_straggler(
            sorted((perf_dir / "opstats").glob("*.trace.json"))
        ) if (args.run_perf and perf_dir) else None
        report = build_report_card(
            args, hw_results, runner_metrics, trace_breakdown,
            device_straggler=device_straggler,
            op_straggler=op_straggler,
        )
        result["report_card"] = report
        print_report_card(report)       # always goes to stdout, even in --report mode

    # Full JSON → stdout in verbose mode, log file in --report mode
    json_str = json.dumps(result, indent=2, sort_keys=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(json_str + "\n")
        print(f"\nFull results saved to : {log_path}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
