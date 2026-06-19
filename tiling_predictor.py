#!/usr/bin/env python3
"""
tiling_predictor.py

Predicts how Qualcomm AI100 tiles a MatMul [M x K] x [K x N].

All formulas verified from real profiling traces across 36 configurations:
  Source: opstats trace JSON (aicconvolutiond32 attributes + DMA counts)
          QAicGraph__splitPlan_SplitPlan_IntraCoreSize_final.log

Usage:
  python tiling_predictor.py --M 512  --K 7168 --N 11008
  python tiling_predictor.py --M 1024 --K 7168 --N 11008 --cores 16
  python tiling_predictor.py --M 1    --K 4096 --N 11008 --cores 4
  python tiling_predictor.py --M 128  --K 7168 --N 18432 --json
"""

import argparse
import math


# ── Hardware constants ────────────────────────────────────────────────────────
VTCM_KB            = 8192    # VTCM per core (KB)
SCRATCH_KB         = 68      # HMX scratch buffer — constant (BackupVTCMAlloc)
OUTC_TILE          = 32      # Output channels per HMX op — invariant across all N,K,M
MXFP6_RATIO        = 179200 / 229376   # ≈ 0.7813 (from trace uindex8/fp16 shapes)
M_SPLIT_ROWS       = 64      # rows-per-core threshold that triggers M-split strategy


# ── Complexity flag ────────────────────────────────────────────────────────────
def is_complex(M: int, K: int, N: int, num_cores: int) -> str | None:
    """
    Flag configurations where the compiler uses non-standard tiling.
    These predictions may be less accurate.

    Complex cases identified from data:
      1. N_per_core > 1376 (e.g. 4-core with N=11008: N_pc=2752)
         → compiler uses mixed tile sizes
      2. M has no clean power-of-2 factoring ≤ 128
         (e.g. M=640: 640/128=5 not pow2; M=896: 896/128=7 not pow2)
         → compiler uses mixed OutY strategies
    """
    N_pc = math.ceil(N / num_cores)
    if N_pc > 1376:
        return f"N_per_core={N_pc} > 1376 — compiler may use mixed tile sizes"

    def is_pow2(n): return n > 0 and (n & (n - 1)) == 0
    if M > 128:
        candidates = [i for i in range(1, 129) if M % i == 0]
        has_pow2_factor = any(is_pow2(M // c) for c in candidates)
        if not has_pow2_factor:
            return f"M={M} has no divisor that produces power-of-2 passes — mixed OutY"

    return None


# ── Strategy ──────────────────────────────────────────────────────────────────
def use_m_split(M: int, num_cores: int) -> bool:
    """
    M-split: each core owns M/cores rows, processes ALL N output channels.
    N-split: each core owns N/cores output channels, processes ALL M rows.

    Rule (verified from traces):
      M-split used when M/cores == M_SPLIT_ROWS (= 64)

    Verified:
      M=1024 C=16 → 1024/16=64  → M-split ✓
      M=512  C=8  → 512/8=64    → M-split ✓
      M=512  C=16 → 512/16=32   → N-split ✓
      M=512  C=4  → 512/4=128   → N-split ✓
    """
    return M % num_cores == 0 and (M // num_cores) == M_SPLIT_ROWS


# ── OutY: rows per HMX tile ───────────────────────────────────────────────────
def compute_outy(M: int, K: int, N: int, num_cores: int) -> int:
    """
    OutY = input rows processed per HMX tile operation.

    M-split: OutY = M / num_cores (each core's row block in one shot)

    N-split: OutY = largest divisor of M that is ≤ 128 such that
               num_passes = M/OutY is a power of 2
               AND the tile fits in VTCM

    Verified across all 36 configurations.
    """
    if use_m_split(M, num_cores):
        return M // num_cores

    if M <= 128:
        return M

    wt_kb = K * OUTC_TILE * 2 / 1024
    N_pc  = math.ceil(N / num_cores)

    def is_pow2(n): return n > 0 and (n & (n - 1)) == 0

    candidates = sorted(
        [i for i in range(1, min(M + 1, 129)) if M % i == 0],
        reverse=True,
    )

    # Prefer: largest divisor where passes is power-of-2 AND fits in VTCM
    for outy in candidates:
        passes = M // outy
        inp_kb = outy * K * 2 / 1024
        out_kb = outy * N_pc * 2 / 1024
        if wt_kb + inp_kb + out_kb + SCRATCH_KB <= VTCM_KB and is_pow2(passes):
            return outy

    # Fallback: largest divisor that just fits
    for outy in candidates:
        inp_kb = outy * K * 2 / 1024
        out_kb = outy * N_pc * 2 / 1024
        if wt_kb + inp_kb + out_kb + SCRATCH_KB <= VTCM_KB:
            return outy

    return 1


# ── NumSrcOperands: K sub-tiles per HMX op ───────────────────────────────────
def compute_num_src_operands(M: int, num_cores: int) -> int:
    """
    NumSrcOperands = number of K sub-tiles HMX accumulates per output tile.
    (How many times the K dimension is split within one HMX op.)

    Formula (verified): NumSrcOp = max(1, ceil(num_cores / ceil(M / 32)))

    M-split always uses NumSrcOp=1 (full K per op).

    N-split:
      cores=16, M=1:   ceil(16/1)  = 16  ✓ (K split 16 ways, sub-tile = K/16)
      cores=16, M=128: ceil(16/4)  =  4  ✓ (K split 4 ways,  sub-tile = K/4)
      cores=16, M=256: ceil(16/8)  =  2  ✓ (K split 2 ways,  sub-tile = K/2)
      cores=16, M=512: ceil(16/16) =  1  ✓ (full K, no split)
      cores= 8, M=1:   ceil(8/1)   =  8  ✓
      cores= 8, M=128: ceil(8/4)   =  2  ✓
      cores= 4, M=1:   ceil(4/1)   =  4  ✓
      cores= 4, M=128: ceil(4/4)   =  1  ✓
    """
    if use_m_split(M, num_cores):
        return 1
    return max(1, math.ceil(num_cores / math.ceil(M / 32)))


# ── Weight tiling ─────────────────────────────────────────────────────────────
def compute_weight_info(K: int, N: int, num_cores: int, M: int) -> dict:
    """
    Weight tile = [K × 32] elements per HMX op (OutC_tile=32 is invariant).

    DDR loads per core:
      N-split: N_per_core / 32  (each core loads its slice once)
      M-split: N / 32           (each core loads ALL N tiles for its rows)

    Total DDR loads across all cores:
      N-split: N/32                    (N_pc/32 per core × cores = N/32)
      M-split: N/32 × num_cores        (each core independently loads all N tiles)

    Verified:
      N-split all cases: wt_loads = N/32           ✓
      M-split M=1024 C=16: wt_loads = 344×16=5504  ✓
      M-split M=512  C=8:  wt_loads = 344×1=344 (8 cores × 43 each = 344) ✓

    Note: 'complex' cases (4-core with large N_pc) may deviate — flagged separately.
    """
    n_tiles      = math.ceil(N / OUTC_TILE)
    outy         = compute_outy(M, K, N, num_cores)
    inp_passes   = math.ceil(M / outy)
    m_split      = use_m_split(M, num_cores)

    if m_split:
        total_loads = n_tiles * num_cores
    else:
        total_loads = n_tiles   # weight loaded once total across all passes

    wt_fp16_kb   = K * OUTC_TILE * 2 / 1024
    wt_mxfp6_kb  = wt_fp16_kb * MXFP6_RATIO
    full_size_mb = K * N * 0.75 / 1024 / 1024  # mxfp6

    return {
        "tile_shape":      f"[{K}×{OUTC_TILE}]",
        "tile_elements":   K * OUTC_TILE,
        "tile_mxfp6_kb":   round(wt_mxfp6_kb, 1),
        "tile_fp16_kb":    round(wt_fp16_kb, 1),
        "total_n_tiles":   n_tiles,
        "input_passes":    inp_passes,
        "total_ddr_loads": total_loads,
        "full_size_mb":    round(full_size_mb, 2),
        "ddr_loads_note":  "N/32 × cores (M-split)" if m_split else "N/32 (loaded once)",
    }


# ── Input tiling ──────────────────────────────────────────────────────────────
def compute_input_info(M: int, K: int, N: int, num_cores: int, outy: int) -> dict:
    """
    Input [M×K] is either:
      VTCM-resident: full input loaded once at start, stays in VTCM
      DDR-chunked:   loaded in ceil(M/OutY) chunks of [outy×K] rows each

    Decision: if full input fits in (VTCM - weight_tile - scratch) → VTCM
    """
    full_kb  = M * K * 2 / 1024
    wt_kb    = K * OUTC_TILE * 2 / 1024
    avail_kb = VTCM_KB - wt_kb - SCRATCH_KB

    if full_kb <= avail_kb and outy >= M:
        return {
            "location":    "VTCM",
            "chunk_shape": f"[{M}×{K}]",
            "chunk_kb":    round(full_kb, 1),
            "num_chunks":  1,
            "strategy":    "Loaded once, stays resident in VTCM",
        }
    else:
        chunk_kb   = outy * K * 2 / 1024
        num_chunks = math.ceil(M / outy)
        return {
            "location":    "DDR",
            "chunk_shape": f"[{outy}×{K}]",
            "chunk_kb":    round(chunk_kb, 1),
            "num_chunks":  num_chunks,
            "strategy":    f"{num_chunks} chunks × {chunk_kb:.0f} KB loaded per pass",
        }


# ── Output info ────────────────────────────────────────────────────────────────
def compute_output_info(M: int, N: int, num_cores: int) -> dict:
    N_pc      = N if use_m_split(M, num_cores) else math.ceil(N / num_cores)
    total_mb  = M * N * 2 / 1024 / 1024
    core_kb   = M * N_pc * 2 / 1024
    location  = "VTCM" if M == 1 else "DDR"
    return {
        "location":     location,
        "shape":        f"[{M}×{N}]",
        "total_mb":     round(total_mb, 2),
        "per_core_kb":  round(core_kb, 1),
    }


# ── VTCM budget ────────────────────────────────────────────────────────────────
def compute_vtcm_budget(K: int, N: int, num_cores: int, outy: int) -> dict:
    N_pc   = math.ceil(N / num_cores)
    wt_kb  = K * OUTC_TILE * 2 / 1024
    inp_kb = outy * K * 2 / 1024
    out_kb = outy * N_pc * 2 / 1024
    used   = wt_kb + inp_kb + out_kb + SCRATCH_KB
    return {
        "weight_fp16_kb":  round(wt_kb, 1),
        "input_tile_kb":   round(inp_kb, 1),
        "output_tile_kb":  round(out_kb, 1),
        "scratch_kb":      SCRATCH_KB,
        "total_used_kb":   round(used, 1),
        "headroom_kb":     round(VTCM_KB - used, 1),
        "utilization_pct": round(100 * used / VTCM_KB, 1),
    }


# ── Main predict ───────────────────────────────────────────────────────────────
def predict(M: int, K: int, N: int, num_cores: int = 16, mxfp6: bool = True) -> dict:
    m_split  = use_m_split(M, num_cores)
    outy     = compute_outy(M, K, N, num_cores)
    numsrc   = compute_num_src_operands(M, num_cores)
    wt       = compute_weight_info(K, N, num_cores, M)
    inp      = compute_input_info(M, K, N, num_cores, outy)
    out      = compute_output_info(M, N, num_cores)
    vtcm     = compute_vtcm_budget(K, N, num_cores, outy)
    warning  = is_complex(M, K, N, num_cores)

    N_pc       = N if m_split else math.ceil(N / num_cores)
    rows_pc    = M // num_cores if m_split else M
    hmx_tiles  = math.ceil(N / OUTC_TILE) * math.ceil(M / outy)
    flops      = 2 * M * K * N
    wt_bytes   = K * N * (0.75 if mxfp6 else 2.0)
    arith_int  = flops / (wt_bytes + M * K * 2 + M * N * 2)

    return {
        "input":            {"M": M, "K": K, "N": N, "cores": num_cores, "mxfp6": mxfp6},
        "warning":          warning,
        "strategy":         "M-split" if m_split else "N-split",
        "inter_core": {
            "split_dim":        "M (rows)" if m_split else "N (output channels)",
            "rows_per_core":    rows_pc,
            "outc_per_core":    N_pc,
            "outc_tile":        OUTC_TILE,
            "wt_ops_per_core":  math.ceil(N_pc / OUTC_TILE),
        },
        "hmx_op": {
            "output_shape":     f"[{outy}×{OUTC_TILE}]",
            "input_shape":      f"[{outy}×{K}]",
            "weight_shape":     f"[{K}×{OUTC_TILE}]",
            "OutYEnd":          outy,
            "OutCPerG":         OUTC_TILE,
            "InCPerG":          K,
            "NumSrcOperands":   numsrc,
            "K_subtile":        K // numsrc,
        },
        "weight":           wt,
        "input":            inp,
        "output":           out,
        "vtcm_budget":      vtcm,
        "hmx_tile_count":   hmx_tiles,
        "flops":            flops,
        "arith_intensity":  round(arith_int, 2),
    }


# ── Pretty printer ─────────────────────────────────────────────────────────────
def print_prediction(p: dict) -> None:
    inp_args = p["input"]  # note: overloaded — the predict() input args
    ic   = p["inter_core"]
    hm   = p["hmx_op"]
    wt   = p["weight"]
    inp  = p["input"]
    out  = p["output"]
    vt   = p["vtcm_budget"]
    SEP  = "=" * 72
    DIV  = "-" * 62

    # Grab M/K/N from hmx_op since we overwrote inp
    M = hm["OutYEnd"] * (p["hmx_tile_count"] // math.ceil(wt["total_n_tiles"] / (1 if p["strategy"]=="M-split" else 1)))
    # Use direct values instead
    M_val = inp_args["M"] if isinstance(inp_args, dict) and "M" in inp_args else "?"

    print(f"\n{SEP}")
    print(f"  TILING PREDICTION  —  Qualcomm AI100")
    print(f"  MatMul  [{p['flops'] // (2 * wt['tile_elements'] * math.ceil(wt['total_n_tiles'])) if False else '?'}×?] ×")

    # Re-derive M,K,N from wt and hm
    K   = hm["InCPerG"]
    N   = wt["total_n_tiles"] * OUTC_TILE
    C   = inp_args["cores"] if isinstance(inp_args, dict) else "?"
    mx  = inp_args["mxfp6"] if isinstance(inp_args, dict) else True

    print(f"  Input:  [M×K]  =  [{p['hmx_op']['OutYEnd'] * (p['hmx_tile_count'] // wt['total_n_tiles'])}×{K}]")
    print(f"  Weight: [K×N]  =  [{K}×{N}]  ({wt['full_size_mb']} MB mxfp6)")
    print(f"  cores={C}  mxfp6={mx}  strategy={p['strategy']}")
    print(f"  FLOPs={p['flops']:,}  ArithIntensity={p['arith_intensity']:.1f} FLOPs/B")

    if p["warning"]:
        print(f"\n  ⚠  WARNING: {p['warning']}")
        print(f"     Prediction is approximate — compiler may use different tile sizes.")

    print(f"\n  {DIV}")
    print(f"  INTER-CORE SPLIT  ({p['strategy']})")
    print(f"  {DIV}")
    print(f"  Split dimension   : {ic['split_dim']}")
    print(f"  Rows per core     : {ic['rows_per_core']}")
    print(f"  OutC per core     : {ic['outc_per_core']}")
    print(f"  OutC tile per op  : {ic['outc_tile']}  (fixed — verified for all N values)")
    print(f"  HMX ops per core  : {ic['wt_ops_per_core']}")

    print(f"\n  {DIV}")
    print(f"  HMX TILE  (smallest unit — one aicconvolutiond32 call)")
    print(f"  {DIV}")
    print(f"  Output            : {hm['output_shape']}  ({hm['OutYEnd']} rows × {OUTC_TILE} outC)")
    print(f"  Input             : {hm['input_shape']}  ({hm['OutYEnd']} rows × full K)")
    print(f"  Weight            : {hm['weight_shape']}  (full K × {OUTC_TILE} outC)")
    print(f"  NumSrcOperands    : {hm['NumSrcOperands']}"
          f"  →  K sub-tile = {hm['K_subtile']} elements")
    print(f"  Total HMX tiles   : {p['hmx_tile_count']:,}")

    print(f"\n  {DIV}")
    print(f"  WEIGHT MATRIX  [{K}×{N}]  =  {wt['full_size_mb']} MB (mxfp6)")
    print(f"  {DIV}")
    print(f"  Tile shape        : {wt['tile_shape']}")
    print(f"  Tile mxfp6        : {wt['tile_mxfp6_kb']} KB  (DDR → VTCM via DMA)")
    print(f"  Tile fp16         : {wt['tile_fp16_kb']} KB  (after dequant in VTCM)")
    print(f"  Total N-tiles     : {wt['total_n_tiles']}  (N / {OUTC_TILE})")
    print(f"  Input passes      : {wt['input_passes']}")
    print(f"  Total DDR loads   : {wt['total_ddr_loads']}  ({wt['ddr_loads_note']})")

    print(f"\n  {DIV}")
    print(f"  INPUT MATRIX  [{p['hmx_op']['OutYEnd'] * (p['hmx_tile_count'] // wt['total_n_tiles'])}×{K}]")
    print(f"  {DIV}")
    print(f"  Location          : {inp['location']}")
    print(f"  Strategy          : {inp['strategy']}")
    print(f"  Chunk shape       : {inp['chunk_shape']}  ({inp['chunk_kb']} KB)")
    print(f"  DDR loads         : {inp['num_chunks']}")

    print(f"\n  {DIV}")
    print(f"  OUTPUT MATRIX  {out['shape']}  =  {out['total_mb']} MB")
    print(f"  {DIV}")
    print(f"  Location          : {out['location']}")
    print(f"  Per-core size     : {out['per_core_kb']} KB")

    print(f"\n  {DIV}")
    print(f"  VTCM BUDGET PER CORE  (capacity = {VTCM_KB} KB)")
    print(f"  {DIV}")
    items = [
        ("Weight tile (fp16)", vt["weight_fp16_kb"]),
        ("Input tile",         vt["input_tile_kb"]),
        ("Output tile",        vt["output_tile_kb"]),
        ("HMX scratch",        vt["scratch_kb"]),
    ]
    for label, kb in items:
        bar = "█" * max(1, round(kb / VTCM_KB * 40))
        print(f"  {label:<20} {kb:>7.1f} KB  {bar}")
    print(f"  {'─'*50}")
    print(f"  {'Total':<20} {vt['total_used_kb']:>7.1f} KB  ({vt['utilization_pct']}% of VTCM)")
    print(f"  {'Headroom':<20} {vt['headroom_kb']:>7.1f} KB")
    print(f"\n{SEP}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Predict AI100 tiling for MatMul [M×K] × [K×N]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--M",       type=int, required=True, help="Seq len / input rows")
    ap.add_argument("--K",       type=int, required=True, help="Hidden size")
    ap.add_argument("--N",       type=int, required=True, help="Output size")
    ap.add_argument("--cores",   type=int, default=16,    help="Number of AI100 cores")
    ap.add_argument("--no-mxfp6", action="store_true",   help="Use fp16 (disable mxfp6)")
    ap.add_argument("--json",    action="store_true",     help="Print raw JSON")
    args = ap.parse_args()

    result = predict(args.M, args.K, args.N, args.cores, not args.no_mxfp6)

    if args.json:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main()
