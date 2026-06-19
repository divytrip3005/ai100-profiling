#!/usr/bin/env python3
"""
matmul_tiler.py

Given a large MatMul [M × K] × [K × N], finds the optimal tile size
(M_b, N_b) for the given K, then decomposes the large matmul into tiles
and estimates performance.

Each tile [M_b × K] × [K × N_b] is compiled as a separate QPC with the
weight slice baked in. At runtime, only the input slice is provided.

Usage:
  # Step 1: Find best tile for K=4096
  python matmul_tiler.py --K 4096 --find-tile

  # Step 2: Decompose large matmul with best tile
  python matmul_tiler.py --M 1024 --K 4096 --N 22016 --M-tile 512 --N-tile 11008

  # Step 1+2 combined (auto-finds tile then decomposes)
  python matmul_tiler.py --M 1024 --K 4096 --N 22016 --auto

  # Specify candidate tile sizes to profile
  python matmul_tiler.py --M 1024 --K 4096 --N 22016 --auto \
    --M-candidates 128,256,512,768,896 \
    --N-candidates 4096,11008,18432
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


BENCHMARK   = Path(__file__).parent / "matmul_microbenchmark.py"
RESULTS_DIR = Path(__file__).parent / "tiler_results"

# Default candidate tile sizes to profile
DEFAULT_M_CANDIDATES = [128, 256, 512, 640, 768, 832, 896, 1024, 1152]
DEFAULT_N_CANDIDATES = [4096, 7168, 11008, 14336, 18432]


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_last_json(txt):
    depth = 0
    for i in range(len(txt) - 1, -1, -1):
        if txt[i] == '}':
            if depth == 0:
                end = i
            depth += 1
        elif txt[i] == '{':
            depth -= 1
            if depth == 0:
                return txt[i:end + 1]
    return None


def get_report_card(log_path):
    if not log_path.exists():
        return {}
    blob = extract_last_json(log_path.read_text())
    if blob:
        try:
            return json.loads(blob).get("report_card", {})
        except Exception:
            pass
    return {}


def get_dequant_and_outc(out_dir):
    """Returns (dequant_loc, OutCPerG) from trace."""
    opstats = out_dir / "perf_dump/opstats"
    if not opstats.exists():
        return "?", None
    files = sorted(opstats.glob("*.trace.json"))
    if not files:
        return "?", None
    mid = files[len(files) // 2]
    try:
        with open(mid) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])

        tid_to_core = {}
        for e in events:
            if e.get("ph") == "M" and e.get("name") == "thread_name":
                m = re.match(r"QAicGraph.*_Core_(\d+)",
                             e.get("args", {}).get("name", ""))
                if m:
                    tid_to_core[e.get("tid")] = int(m.group(1))

        dq_loc = "?"
        outc   = None
        for e in events:
            if e.get("ph") != "X":
                continue
            if e.get("name") == "blockdequantize_mxfp6" and dq_loc == "?":
                op0 = e.get("args", {}).get("opOperand0", "")
                m = re.search(r"(TCM|DDR)", op0)
                dq_loc = m.group(1) if m else "?"
            if (e.get("name") == "aicconvolutiond32"
                    and tid_to_core.get(e.get("tid")) == 0
                    and outc is None):
                attrs = e.get("args", {}).get("opAttributes", "")
                m = re.search(r"OutCPerG:\s*(\d+)", attrs)
                if m:
                    outc = int(m.group(1))
        return dq_loc, outc
    except Exception:
        return "?", None


# ── Run one benchmark ─────────────────────────────────────────────────────────
def run_benchmark(M, K, N, cores, device_group, mxfp6, out_dir, log_path):
    """Compile + profile one (M, K, N) tile. Returns report_card."""

    # reuse existing result if available
    rc = get_report_card(log_path)
    if rc:
        return rc

    cmd = [
        sys.executable, str(BENCHMARK),
        f"--hidden-size={K}",
        f"--out-size={N}",
        f"--seq-len={M}",
        f"--compile-num-cores={cores}",
        f"--device-group={device_group}",
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

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = proc.stdout + proc.stderr
    log_path.write_text(output)

    blob = extract_last_json(output)
    if blob:
        try:
            return json.loads(blob).get("report_card", {})
        except Exception:
            pass
    return {}


# ── Find best tile ────────────────────────────────────────────────────────────
def find_best_tile(K, M_candidates, N_candidates, cores, device_group, mxfp6):
    """Profile all (M_b, N_b) candidates and return ranked results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total   = len(M_candidates) * len(N_candidates)
    done    = 0
    results = []

    print(f"\n  Profiling {total} tile candidates (K={K})...")
    print(f"  {'M_b':>6} {'N_b':>7} {'ExecUs':>9} {'TFLOPS':>8} "
          f"{'HMX%':>6} {'OutCPerG':>9} {'Dequant':>8}  Status")
    print("  " + "-" * 75)

    for M_b in M_candidates:
        for N_b in N_candidates:
            done += 1
            tag     = f"tile_K{K}_M{M_b}_N{N_b}"
            out_dir = RESULTS_DIR / tag
            log     = RESULTS_DIR / f"{tag}.log"

            print(f"  [{done:>3}/{total}] M={M_b} N={N_b}...",
                  end=" ", flush=True)

            rc = run_benchmark(M_b, K, N_b, cores, device_group,
                               mxfp6, out_dir, log)
            if not rc:
                print("FAILED")
                continue

            t      = rc.get("exec_time_us", 0)
            hmx    = rc.get("hmx_active_pct", 0)
            ddr    = rc.get("ddr_traffic_mb", 0)
            bn     = rc.get("bottleneck", "?")
            flops  = 2 * M_b * K * N_b
            tflops = flops / t / 1e6 if t else 0

            dq, outc = get_dequant_and_outc(out_dir)
            outc_s   = str(outc) if outc else "?"

            is_cliff = (dq == "DDR")
            is_good  = (not is_cliff and hmx >= 50 and tflops >= 15)
            status   = "⚠ CLIFF" if is_cliff else ("✓ GOOD" if is_good else "~ OK")

            results.append({
                "M_b": M_b, "N_b": N_b, "K": K,
                "exec_us": t, "tflops": tflops,
                "hmx": hmx, "ddr_mb": ddr,
                "bottleneck": bn,
                "dq": dq, "outc": outc,
                "status": status,
            })

            marker = "❌" if is_cliff else ("✓" if is_good else " ")
            print(f"done  {t:>9.0f}µs {tflops:>8.1f}T {hmx:>6.0f}% "
                  f"{outc_s:>9} {dq:>8}  {status}")

    return sorted(results, key=lambda x: -x["tflops"])


# ── Decompose large matmul ────────────────────────────────────────────────────
def decompose(M, K, N, M_b, N_b, tile_exec_us, devices):
    """Show decomposition plan and time estimates."""
    import math
    tiles_M = math.ceil(M / M_b)
    tiles_N = math.ceil(N / N_b)
    total_tiles = tiles_M * tiles_N

    # last tile may be smaller (remainder)
    M_rem = M % M_b if M % M_b != 0 else M_b
    N_rem = N % N_b if N % N_b != 0 else N_b

    t_seq    = total_tiles * tile_exec_us
    t_par    = math.ceil(total_tiles / devices) * tile_exec_us

    print(f"\n{'='*65}")
    print(f"  DECOMPOSITION PLAN")
    print(f"  Large matmul: [{M} × {K}] × [{K} × {N}]")
    print(f"  Tile size:    [{M_b} × {K}] × [{K} × {N_b}]")
    print(f"{'='*65}")

    print(f"\n  Split:")
    print(f"    M dimension: {M} / {M_b} = {tiles_M} tiles"
          + (f"  (last tile: {M_rem} rows)" if M_rem != M_b else ""))
    print(f"    N dimension: {N} / {N_b} = {tiles_N} tiles"
          + (f"  (last tile: {N_rem} cols)" if N_rem != N_b else ""))
    print(f"    Total tiles: {tiles_M} × {tiles_N} = {total_tiles}")

    print(f"\n  Tile grid:")
    for i in range(tiles_M):
        row_start = i * M_b
        row_end   = min(row_start + M_b, M)
        row_str   = f"rows {row_start}-{row_end-1} ({row_end-row_start})"
        for j in range(tiles_N):
            col_start = j * N_b
            col_end   = min(col_start + N_b, N)
            col_str   = f"cols {col_start}-{col_end-1} ({col_end-col_start})"
            shape_in  = f"[{row_end-row_start} × {K}]"
            shape_w   = f"[{K} × {col_end-col_start}]"
            print(f"    T({i},{j}): {shape_in} × {shape_w}"
                  f"  ← {row_str}, {col_str}")

    print(f"\n  Runtime per tile: {tile_exec_us:.0f} µs")
    print(f"\n  ─── Time estimates ───────────────────────────────────")
    print(f"  Sequential ({total_tiles} tiles one-by-one):"
          f"  {t_seq:.0f} µs = {t_seq/1000:.1f} ms")
    print(f"  Parallel   ({devices} devices):           "
          f"  {t_par:.0f} µs = {t_par/1000:.1f} ms"
          f"  ({total_tiles} tiles / {devices} devices = "
          f"{math.ceil(total_tiles/devices)} rounds)")

    print(f"\n  ─── Compiled QPCs needed ────────────────────────────")
    print(f"  One QPC per unique tile shape:")
    shapes = set()
    for i in range(tiles_M):
        m_actual = min(M_b, M - i * M_b)
        for j in range(tiles_N):
            n_actual = min(N_b, N - j * N_b)
            shapes.add((m_actual, K, n_actual))
    for s in sorted(shapes):
        count = sum(1 for i in range(tiles_M)
                    for j in range(tiles_N)
                    if min(M_b, M-i*M_b) == s[0]
                    and min(N_b, N-j*N_b) == s[2])
        print(f"    [{s[0]} × {s[1]}] × [{s[1]} × {s[2]}]"
              f"  →  {count} tiles use this QPC")

    print(f"\n  ─── Runtime input shape per device ──────────────────")
    print(f"  Each device receives: [1 × M_b × K] = "
          f"[1 × {M_b} × {K}]  ({M_b*K*2//1024} KB fp16)")
    print(f"  Weight baked into QPC at compile time")
    print(f"  Output per device:    [1 × {M_b} × {N_b}]  "
          f"({M_b*N_b*2//1024} KB fp16)")

    return {
        "tiles_M": tiles_M, "tiles_N": tiles_N,
        "total_tiles": total_tiles,
        "t_sequential_us": t_seq,
        "t_parallel_us": t_par,
        "unique_qpcs": len(shapes),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Find optimal tile for large MatMul and decompose it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--K",            type=int, required=True,
                    help="Hidden size (fixed dimension)")
    ap.add_argument("--M",            type=int, default=None,
                    help="Large M to decompose")
    ap.add_argument("--N",            type=int, default=None,
                    help="Large N to decompose")
    ap.add_argument("--M-tile",       type=int, default=None,
                    help="M tile size (skip profiling, use this directly)")
    ap.add_argument("--N-tile",       type=int, default=None,
                    help="N tile size (skip profiling, use this directly)")
    ap.add_argument("--M-candidates", type=str, default=None,
                    help="Comma-separated M tile sizes to profile")
    ap.add_argument("--N-candidates", type=str, default=None,
                    help="Comma-separated N tile sizes to profile")
    ap.add_argument("--cores",        type=int, default=16)
    ap.add_argument("--device-group", type=str, default="[0]")
    ap.add_argument("--devices",      type=int, default=4,
                    help="Number of devices available for parallel execution")
    ap.add_argument("--no-mxfp6",    action="store_true")
    ap.add_argument("--find-tile",   action="store_true",
                    help="Only profile tile candidates, skip decomposition")
    ap.add_argument("--auto",        action="store_true",
                    help="Profile tiles then decompose automatically")
    ap.add_argument("--top",         type=int, default=5,
                    help="Show top N tile candidates")
    args = ap.parse_args()

    M_cands = ([int(x) for x in args.M_candidates.split(",")]
               if args.M_candidates else DEFAULT_M_CANDIDATES)
    N_cands = ([int(x) for x in args.N_candidates.split(",")]
               if args.N_candidates else DEFAULT_N_CANDIDATES)
    mxfp6   = not args.no_mxfp6

    # ── mode: find-tile or auto (profile) ────────────────────────────────────
    ranked = []
    if args.find_tile or args.auto or (args.M_tile is None and args.N_tile is None):
        print(f"\nProfiling tile candidates for K={args.K}...")
        ranked = find_best_tile(args.K, M_cands, N_cands,
                                args.cores, args.device_group, mxfp6)

        print(f"\n{'='*65}")
        print(f"  TOP {args.top} TILE CANDIDATES  (ranked by TFLOPS)")
        print(f"{'='*65}")
        print(f"  {'Rank':>5} {'M_b':>6} {'N_b':>7} {'TFLOPS':>8} "
              f"{'HMX%':>6} {'ExecUs':>8} {'Status'}")
        print("  " + "-" * 55)
        for i, r in enumerate(ranked[:args.top], 1):
            marker = "❌" if "CLIFF" in r["status"] else ("✓" if "GOOD" in r["status"] else " ")
            print(f"  {i:>5} {marker} M={r['M_b']:<5} N={r['N_b']:<6} "
                  f"{r['tflops']:>8.1f}T {r['hmx']:>6.0f}% "
                  f"{r['exec_us']:>8.0f}  {r['status']}")

        # pick best
        best = ranked[0] if ranked else None
        if best:
            print(f"\n  Best tile: M_b={best['M_b']}, N_b={best['N_b']}, K={args.K}")
            print(f"  → {best['tflops']:.1f} TFLOPS, {best['hmx']:.0f}% HMX, "
                  f"{best['exec_us']:.0f}µs, dequant={best['dq']}")

        if args.find_tile:
            return

        # use best for decomposition
        if args.M_tile is None and best:
            args.M_tile = best["M_b"]
        if args.N_tile is None and best:
            args.N_tile = best["N_b"]

    # ── mode: decompose ───────────────────────────────────────────────────────
    if args.M is None or args.N is None:
        print("\nProvide --M and --N to decompose a large matmul.")
        return
    if args.M_tile is None or args.N_tile is None:
        print("\nProvide --M-tile and --N-tile (or use --auto to profile first).")
        return

    # get tile exec time
    tile_exec_us = None
    # from profiling results
    for r in ranked:
        if r["M_b"] == args.M_tile and r["N_b"] == args.N_tile:
            tile_exec_us = r["exec_us"]
            break
    # from existing benchmark result
    if tile_exec_us is None:
        tag = f"tile_K{args.K}_M{args.M_tile}_N{args.N_tile}"
        rc  = get_report_card(RESULTS_DIR / f"{tag}.log")
        if rc:
            tile_exec_us = rc.get("exec_time_us", 0)
    # from original results dir
    if tile_exec_us is None:
        orig = Path(__file__).parent / "results"
        for candidate in [
            f"h{args.K}_o{args.N_tile}_s{args.M_tile}",
        ]:
            rc = get_report_card(orig / f"{candidate}.log")
            if rc:
                tile_exec_us = rc.get("exec_time_us", 0)
                break
    if not tile_exec_us:
        print(f"\nNo profiling data for tile M={args.M_tile}, N={args.N_tile}, K={args.K}")
        print(f"Run with --auto or --find-tile first.")
        return

    decompose(args.M, args.K, args.N,
              args.M_tile, args.N_tile,
              tile_exec_us, args.devices)


if __name__ == "__main__":
    main()
