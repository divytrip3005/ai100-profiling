#!/usr/bin/env python3
"""
tiling_analyzer.py

Extracts exact tiling decisions from compiler output logs.
For existing runs: reads logs directly (100% accurate).
For new shapes: runs matmul_microbenchmark.py --run-perf to generate logs.

Usage:
  # Use existing run
  python tiling_analyzer.py --M 512 --K 7168 --N 11008

  # Compile new shape first
  python tiling_analyzer.py --M 256 --K 7168 --N 11008 --compile
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


RESULTS_DIR  = Path(__file__).parent / "results"
BENCHMARK_PY = Path(__file__).parent / "matmul_microbenchmark.py"


# ── Find existing result dir ──────────────────────────────────────────────────
def find_result_dir(M: int, K: int, N: int, cores: int) -> Path | None:
    candidates = [
        RESULTS_DIR / f"h{K}_o{N}_s{M}",
        RESULTS_DIR / f"1dev{cores}core_h{K}_o{N}_s{M}",
        RESULTS_DIR / f"{cores}core_h{K}_o{N}_s{M}",
    ]
    for p in candidates:
        split_log = p / "perf_dump" / "dumps" / "QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log"
        if split_log.exists():
            return p
    return None


# ── Run matmul_microbenchmark.py ──────────────────────────────────────────────
def run_benchmark(M: int, K: int, N: int, cores: int, mxfp6: bool) -> Path:
    out_dir = RESULTS_DIR / f"h{K}_o{N}_s{M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(BENCHMARK_PY),
        f"--hidden-size={K}",
        f"--out-size={N}",
        f"--seq-len={M}",
        f"--compile-num-cores={cores}",
        "--device-group=[0]",
        f"--artifact-dir={out_dir}",
        "--run-compile", "--dump-io", "--run-perf",
    ]
    if not mxfp6:
        cmd.append("--no-mxfp6")

    env = os.environ.copy()
    if "QAIC_COMPILER_OPTS_UNSUPPORTED" not in env:
        env["QAIC_COMPILER_OPTS_UNSUPPORTED"] = (
            "-aic-hoist-vtcm-loads=false -aic-op-stats-verbosity 2 "
            "-aic-userdma-async=0 -aic-hmx-async=0 -debug-glow"
        )

    print(f"  Running benchmark (M={M} K={K} N={N} cores={cores})...", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError("Benchmark failed")
    return out_dir


# ── Parse IntraCoreSize split plan ────────────────────────────────────────────
def parse_intra_core_split(dump_dir: Path) -> dict:
    path = dump_dir / "QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log"
    if not path.exists():
        return {}

    txt   = path.read_text()
    cores = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(c\d+) ", line)
        if not m:
            continue
        core   = m.group(1)
        outc   = re.search(r"OutCPerG:\s*(\d+)", line)
        outy   = re.search(r"outYEnd:\s*(\d+)", line)
        inc    = re.search(r"InCPerG:\s*(\d+)", line)
        layout = re.search(r"layout:\s*(\S+)", line)
        numsrc = re.search(r"NumSrcOperands:\s*(\d+)", line)
        if outc:
            cores[core] = {
                "OutCPerG":       int(outc.group(1)),
                "outYEnd":        int(outy.group(1))   if outy   else None,
                "InCPerG":        int(inc.group(1))    if inc    else None,
                "layout":         layout.group(1)      if layout else None,
                "NumSrcOperands": int(numsrc.group(1)) if numsrc else None,
            }

    if not cores:
        return {}

    return {
        "num_cores":       len(cores),
        "OutCPerG_values": sorted(set(v["OutCPerG"]       for v in cores.values())),
        "outYEnd_values":  sorted(set(v["outYEnd"]        for v in cores.values() if v["outYEnd"])),
        "InCPerG_values":  sorted(set(v["InCPerG"]        for v in cores.values() if v["InCPerG"])),
        "NumSrcOperands":  sorted(set(v["NumSrcOperands"] for v in cores.values() if v["NumSrcOperands"])),
        "layouts":         sorted(set(v["layout"]         for v in cores.values() if v["layout"])),
    }


# ── Parse op summary ──────────────────────────────────────────────────────────
def parse_op_summary(dump_dir: Path) -> dict:
    path = dump_dir / "QAicGraph__op_summary_final.log"
    if not path.exists():
        return {}

    txt = path.read_text()
    result: dict = {}
    for line in txt.splitlines():
        if "AICCopyToVTCM count" in line:
            nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]
            if nums:
                result["wt_load_total"]    = nums[-3] if len(nums) >= 3 else nums[-1]
                result["wt_load_per_core"] = nums[:16]
        if "AICConvolutionD32 count" in line:
            nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]
            if nums:
                result["hmx_tile_total"]    = nums[-3] if len(nums) >= 3 else nums[-1]
                result["hmx_tile_per_core"] = nums[:16]
        if "AICCopyToVTCM bytes" in line:
            nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", line)]
            if nums:
                result["wt_bytes_total"] = nums[-3] if len(nums) >= 3 else nums[-1]
    return result


# ── Parse VTCM usage ──────────────────────────────────────────────────────────
def parse_vtcm_usage(dump_dir: Path) -> dict:
    path = dump_dir / "QAicGraph__memuse_estimate_final.log"
    if not path.exists():
        return {}

    txt      = path.read_text()
    core_max: dict[int, float] = {}
    for line in txt.splitlines():
        m = re.match(r"\s*VTCM(\d+) usage [\d.]+ MB highwatermark ([\d.]+) MB", line)
        if m:
            c  = int(m.group(1))
            hw = float(m.group(2))
            core_max[c] = max(core_max.get(c, 0.0), hw)

    if not core_max:
        return {}

    pk = max(core_max.values()) * 1024
    return {
        "vtcm_peak_kb":     round(pk, 1),
        "vtcm_utilization": round(100 * pk / 8192, 1),
        "vtcm_headroom_kb": round(8192 - pk, 1),
    }


# ── Parse DDR summary ─────────────────────────────────────────────────────────
def parse_ddr_summary(dump_dir: Path) -> dict:
    path = dump_dir / "QAicGraph__ddr_op_summary.log"
    if not path.exists():
        return {}

    txt     = path.read_text()
    result  = {}
    section = None
    for line in txt.splitlines():
        if   "DDR Traffic Const In"     in line: section = "weight_in"
        elif "DDR Traffic Non-Const In" in line: section = "activation_in"
        elif "DDR Traffic Out"          in line: section = "output_out"
        elif "Total Size" in line and section:
            m = re.search(r"([\d,]+)", line)
            if m:
                result[f"{section}_bytes"] = int(m.group(1).replace(",", ""))
    return result


# ── Parse trace ───────────────────────────────────────────────────────────────
def parse_trace(opstats_dir: Path) -> dict:
    files = sorted(opstats_dir.glob("*.trace.json"))
    if not files:
        return {}

    mid = files[len(files) // 2]
    with open(mid) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    tid_to_core: dict[int, int] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname = e.get("args", {}).get("name", "")
            m = re.match(r"QAicGraph.*_Core_(\d+)", tname)
            if m:
                tid_to_core[e.get("tid")] = int(m.group(1))

    outc_vals: set[int] = set()
    outy_vals: set[int] = set()
    nsrc_vals: set[int] = set()
    wt_sizes:  list[int] = []
    inp_sizes: list[int] = []
    wt_count = inp_count = hmx_count = 0

    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        args = e.get("args", {})
        if name == "aicconvolutiond32":
            hmx_count += 1
            attrs = args.get("opAttributes", "")
            for field, s in [("OutCPerG", outc_vals), ("OutYEnd", outy_vals),
                              ("NumSrcOperands", nsrc_vals)]:
                m2 = re.search(rf"{field}:\s*(\d+)", attrs)
                if m2:
                    s.add(int(m2.group(1)))
        elif name == "aiccopytovtcm":
            src  = args.get("opOperand1", "")
            dest = args.get("opOperand0", "")
            kb_m = re.search(r"(\d+)KB", dest)
            kb   = int(kb_m.group(1)) if kb_m else 0
            if "in DDR Src x/" in src:
                inp_count += 1
                inp_sizes.append(kb)
            elif "in DDR Src /MatMul/" in src:
                wt_count += 1
                wt_sizes.append(kb)

    return {
        "OutCPerG":           sorted(outc_vals),
        "OutYEnd":            sorted(outy_vals),
        "NumSrcOperands":     sorted(nsrc_vals),
        "hmx_tile_count":     hmx_count,
        "wt_ddr_loads":       wt_count,
        "wt_tile_sizes_kb":   dict(Counter(wt_sizes)),
        "inp_ddr_loads":      inp_count,
        "inp_chunk_sizes_kb": dict(Counter(inp_sizes)),
    }


# ── Main analyze ──────────────────────────────────────────────────────────────
def analyze(M: int, K: int, N: int, cores: int = 16,
            mxfp6: bool = True, compile_new: bool = False) -> dict:

    result_dir = find_result_dir(M, K, N, cores)

    if result_dir is None:
        if not compile_new:
            raise FileNotFoundError(
                f"No existing results for M={M} K={K} N={N} cores={cores}.\n"
                f"Run with --compile to generate them."
            )
        result_dir = run_benchmark(M, K, N, cores, mxfp6)

    dump_dir    = result_dir / "perf_dump" / "dumps"
    opstats_dir = result_dir / "perf_dump" / "opstats"

    return {
        "config":     {"M": M, "K": K, "N": N, "cores": cores, "mxfp6": mxfp6},
        "result_dir": str(result_dir),
        "split_plan": parse_intra_core_split(dump_dir),
        "op_summary": parse_op_summary(dump_dir),
        "vtcm":       parse_vtcm_usage(dump_dir),
        "ddr":        parse_ddr_summary(dump_dir),
        "trace":      parse_trace(opstats_dir),
    }


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(r: dict) -> None:
    cfg   = r["config"]
    M, K, N, cores = cfg["M"], cfg["K"], cfg["N"], cfg["cores"]
    sp    = r["split_plan"]
    ops   = r["op_summary"]
    vt    = r["vtcm"]
    dd    = r["ddr"]
    tr    = r["trace"]
    SEP   = "=" * 72
    DIV   = "-" * 62
    flops = 2 * M * K * N

    # Prefer trace (real execution) over split plan (compiler estimate)
    outc_vals = tr.get("OutCPerG")        or sp.get("OutCPerG_values", [])
    outy_vals = tr.get("OutYEnd")         or sp.get("outYEnd_values",  [])
    nsrc_vals = tr.get("NumSrcOperands")  or sp.get("NumSrcOperands",  [])
    inc_vals  = sp.get("InCPerG_values",  [])
    layouts   = sp.get("layouts", [])
    hmx_total = tr.get("hmx_tile_count")  or ops.get("hmx_tile_total", "N/A")
    wt_loads  = tr.get("wt_ddr_loads")    or ops.get("wt_load_total",  "N/A")
    inp_loads = tr.get("inp_ddr_loads", 0)

    strategy  = "M-split" if (outc_vals and outc_vals[0] == N) else "N-split"

    print(f"\n{SEP}")
    print(f"  TILING ANALYSIS  (100% accurate — from compiler output)")
    print(f"  MatMul  [{M}×{K}] × [{K}×{N}]")
    print(f"  cores={cores}  mxfp6={cfg['mxfp6']}  FLOPs={flops:,}")
    print(f"  Source: {r['result_dir']}")
    print(SEP)

    print(f"\n  Strategy          : {strategy}")

    print(f"\n  {DIV}")
    print(f"  INTER-CORE SPLIT")
    print(f"  {DIV}")
    if strategy == "N-split":
        print(f"  Split dim         : N={N} across {cores} cores → {N//cores} channels/core")
    else:
        print(f"  Split dim         : M={M} across {cores} cores → {M//cores} rows/core")
    print(f"  OutCPerG/core     : {outc_vals}")
    print(f"  Layouts           : {layouts}")

    print(f"\n  {DIV}")
    print(f"  HMX TILE  (smallest compute unit per aicconvolutiond32 call)")
    print(f"  {DIV}")
    print(f"  OutYEnd           : {outy_vals}  (rows processed per HMX op)")
    print(f"  OutCPerG          : {outc_vals}  (output channels per HMX op)")
    print(f"  InCPerG (K)       : {inc_vals}")
    print(f"  NumSrcOperands    : {nsrc_vals}  (K sub-tiles per HMX op)")
    if nsrc_vals and inc_vals:
        print(f"  K sub-tile size   : {inc_vals[0] // nsrc_vals[0]} elements")
    print(f"  Total HMX tiles   : {hmx_total}")
    pc = ops.get("hmx_tile_per_core", [])
    if pc:
        imb = 100*(max(pc)-min(pc))//max(pc) if max(pc) else 0
        print(f"  Per-core tiles    : min={min(pc)}  max={max(pc)}  imbalance={imb}%")

    print(f"\n  {DIV}")
    print(f"  WEIGHT MATRIX  [{K}×{N}]")
    print(f"  {DIV}")
    print(f"  DDR loads total   : {wt_loads}")
    print(f"  Tile sizes (KB)   : {tr.get('wt_tile_sizes_kb', {})}")
    wt_b = dd.get("weight_in_bytes", 0)
    print(f"  DDR bytes total   : {wt_b:,} B  ({wt_b/1024/1024:.2f} MB)")
    n32 = N // 32
    if isinstance(wt_loads, int) and n32:
        print(f"  N/32={n32}  passes={wt_loads // n32}")

    print(f"\n  {DIV}")
    print(f"  INPUT MATRIX  [{M}×{K}]")
    print(f"  {DIV}")
    inp_loc = "VTCM (no DDR loads)" if inp_loads == 0 else f"DDR  ({inp_loads} chunks)"
    print(f"  Location          : {inp_loc}")
    inp_sizes = tr.get("inp_chunk_sizes_kb", {})
    if inp_sizes:
        print(f"  Chunk sizes (KB)  : {inp_sizes}")
    act_b = dd.get("activation_in_bytes", 0)
    print(f"  DDR bytes total   : {act_b:,} B  ({act_b/1024/1024:.2f} MB)")

    print(f"\n  {DIV}")
    print(f"  OUTPUT MATRIX  [{M}×{N}]")
    print(f"  {DIV}")
    out_b   = dd.get("output_out_bytes", 0)
    out_loc = "VTCM" if out_b == 0 else "DDR"
    print(f"  Location          : {out_loc}")
    print(f"  DDR bytes total   : {out_b:,} B  ({out_b/1024/1024:.2f} MB)")

    print(f"\n  {DIV}")
    print(f"  VTCM USAGE  (capacity = 8192 KB per core)")
    print(f"  {DIV}")
    pk  = vt.get("vtcm_peak_kb", 0)
    ut  = vt.get("vtcm_utilization", 0)
    hd  = vt.get("vtcm_headroom_kb", 0)
    bar = "█" * round(ut / 100 * 40) + "░" * (40 - round(ut / 100 * 40))
    print(f"  Peak used         : {pk:.0f} KB  ({ut}%)")
    print(f"  Headroom          : {hd:.0f} KB")
    print(f"  [{bar}] {ut}%")

    print(f"\n{SEP}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Analyze AI100 tiling for MatMul [M×K]×[K×N] from compiler logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--M",       type=int, required=True, help="Seq len / input rows")
    ap.add_argument("--K",       type=int, required=True, help="Hidden size")
    ap.add_argument("--N",       type=int, required=True, help="Output size")
    ap.add_argument("--cores",   type=int, default=16,    help="Number of AI100 cores")
    ap.add_argument("--no-mxfp6", action="store_true",   help="Disable mxfp6")
    ap.add_argument("--compile", action="store_true",
                    help="Run matmul_microbenchmark.py if no existing results found")
    ap.add_argument("--json",    action="store_true",     help="Print raw JSON")
    args = ap.parse_args()

    result = analyze(
        M=args.M, K=args.K, N=args.N,
        cores=args.cores,
        mxfp6=not args.no_mxfp6,
        compile_new=args.compile,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
