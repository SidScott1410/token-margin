"""
Asking price vs measured cost floor. See HYPOTHESIS.md for the model and the
pre-registered hypotheses; this file only computes.

    python margin.py                 # writes results/*.csv and prints the headline table
    python margin.py --u 0.9 --batch 32

Inputs: data/*.jsonl (reckon schema 5), data/tokenprice-latest.json, prices.yaml.
No network. No fitted parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_MAP = {  # corpus model -> tokenprice normalized keys that sell the same architecture
    "llama3-8b": ["llama-3-1-8b-instruct"],
    "qwen3-30b-a3b": ["qwen3-30b-a3b", "qwen3-30b-a3b-instruct-2507"],
}
HEADLINE = {"batch": 32, "prompt_lo": 1500, "prompt_hi": 2500, "output": 256}


def load_rates() -> dict:
    """Tiny YAML subset reader: the file is flat and hand-written; avoids a dependency."""
    rates = {}
    for line in (ROOT / "prices.yaml").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line or "{" not in line:
            continue
        name, rest = line.split(":", 1)
        body = rest.strip().strip("{}")
        kv = {}
        for part in body.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip()] = v.strip().strip('"')
        if "rate" in kv:
            rates[name.strip()] = {"rate": float(kv["rate"]), "provenance": kv.get("provenance"), "as_of": kv.get("as_of")}
    return rates


def load_cells() -> list[dict]:
    cells = []
    for f in sorted((ROOT / "data").glob("*.jsonl")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                r["file"] = f.name
                r["precision"] = "fp8" if "fp8" in f.name else "bf16"
                cells.append(r)
    return cells


def load_sellers() -> dict[str, list[dict]]:
    d = json.load((ROOT / "data" / "tokenprice-latest.json").open())
    by: dict[str, list[dict]] = defaultdict(list)
    for r in d["records"]:
        if r["provider"] == "openrouter":
            continue
        by[r["model"]].append(r)
    return by, d["read_at"]


def floor_usd_per_m(cell: dict, rate: float, u: float) -> float:
    P, O, B = cell["avg_prompt_tokens"], cell["avg_output_tokens"], cell["batch"]
    t_ms = cell["measured_ttft_ms"] + cell["measured_tpot_ms"] * (O - 1)
    cost_req = rate / 3600 * t_ms / 1000 / B / u
    return cost_req / (P + O) * 1e6


def breakeven(cells: list[dict], model: str, silicon: str, rate: float, u: float, price: float, O: int) -> dict:
    """What would make `price` break even on this silicon?

    Fits time_per_batch = a + b*B over the measured batches at the headline mix (the
    decode roofline is linear in B: weights once, KV per sequence), then solves for the
    batch where floor == price. Beyond the measured range it is an extrapolation and is
    labeled as such. Also returns the GPU rate at which B=32 would break even.
    """
    pts = [(c["batch"], c["measured_ttft_ms"] + c["measured_tpot_ms"] * (O - 1), c["avg_prompt_tokens"])
           for c in cells if c["model_name"] == model and c["silicon"] == silicon and c["precision"] == "bf16"
           and c["avg_output_tokens"] == O and HEADLINE["prompt_lo"] <= c["avg_prompt_tokens"] <= HEADLINE["prompt_hi"]]
    if len(pts) < 3:
        return {}
    xs = [b for b, _, _ in pts]
    ys = [t for _, t, _ in pts]
    P = pts[0][2]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    icpt = my - slope * mx
    # floor(B) = rate/3600 * (icpt + slope*B)/1000 / B / u / (P+O) * 1e6 ; solve floor(B) = price
    k = rate / 3600 / 1000 / u / (P + O) * 1e6
    denom = price - k * slope
    B_star = (k * icpt) / denom if denom > 0 else float("inf")
    t32 = icpt + slope * 32
    rate32 = price * 32 * u * (P + O) / 1e6 * 3600 * 1000 / t32
    return {"fit_ms_per_batch": f"{icpt:.0f} + {slope:.1f}*B", "max_measured_batch": max(xs),
            "breakeven_batch": round(B_star, 0) if B_star != float("inf") else "never (price below per-sequence asymptote)",
            "extrapolated": B_star > max(xs), "breakeven_gpu_rate_at_b32": round(rate32, 2)}


def seller_usd_per_m(s: dict, P: int, O: int) -> float:
    return (P * s["input_per_m"] + O * s["output_per_m"]) / (P + O)


def main(u: float, batch: int) -> None:
    rates = load_rates()
    cells = load_cells()
    sellers, read_at = load_sellers()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # 1. floors: every cell
    floor_rows = []
    for c in cells:
        r = rates.get(c["silicon"])
        if not r:
            continue
        floor_rows.append({
            "model": c["model_name"], "silicon": c["silicon"], "precision": c["precision"], "batch": c["batch"],
            "prompt": c["avg_prompt_tokens"], "output": c["avg_output_tokens"],
            "ttft_ms": round(c["measured_ttft_ms"], 1), "tpot_ms": round(c["measured_tpot_ms"], 3),
            "gpu_rate": r["rate"], "rate_provenance": r["provenance"],
            "floor_usd_per_m_u30": round(floor_usd_per_m(c, r["rate"], 0.3), 4),
            "floor_usd_per_m_u60": round(floor_usd_per_m(c, r["rate"], 0.6), 4),
            "floor_usd_per_m_u90": round(floor_usd_per_m(c, r["rate"], 0.9), 4),
        })
    _write(out / "floors.csv", floor_rows)

    # 2. headline cell per silicon, then sellers against it
    head = [f for f in floor_rows if f["batch"] == batch and HEADLINE["prompt_lo"] <= f["prompt"] <= HEADLINE["prompt_hi"]
            and f["output"] == HEADLINE["output"]]
    ukey = f"floor_usd_per_m_u{int(u * 100)}"
    margin_rows = []
    summary = []
    for model, keys in MODEL_MAP.items():
        fl = [f for f in head if f["model"] == model]
        if not fl:
            continue
        cheapest = min(fl, key=lambda f: f[ukey])
        dearest = max(fl, key=lambda f: f[ukey])
        P, O = cheapest["prompt"], cheapest["output"]
        sells = [s for k in keys for s in sellers.get(k, [])]
        # collapse each provider to its cheapest listing for this architecture
        best: dict[str, dict] = {}
        for s in sells:
            p = seller_usd_per_m(s, P, O)
            if s["provider"] not in best or p < seller_usd_per_m(best[s["provider"]], P, O):
                best[s["provider"]] = s
        prices = []
        for prov, s in sorted(best.items(), key=lambda kv: seller_usd_per_m(kv[1], P, O)):
            p = seller_usd_per_m(s, P, O)
            prices.append(p)
            be = breakeven(cells, model, cheapest["silicon"], cheapest["gpu_rate"], u, p, O) if p < cheapest[ukey] else {}
            margin_rows.append({
                "model": model, "provider": prov, "model_raw": s["model_raw"],
                "quantization": (s.get("extra") or {}).get("quantization") or "",
                "input_per_m": s["input_per_m"], "output_per_m": s["output_per_m"],
                "seller_usd_per_m_at_mix": round(p, 4),
                "floor_cheapest_silicon": cheapest["silicon"], "floor_usd_per_m": cheapest[ukey],
                "implied_margin_vs_cheapest": round(1 - cheapest[ukey] / p, 3) if p else None,
                "floor_dearest_silicon": dearest["silicon"],
                "implied_margin_vs_dearest": round(1 - dearest[ukey] / p, 3) if p else None,
                "below_floor_u90": seller_usd_per_m(s, P, O) < cheapest["floor_usd_per_m_u90"],
                "breakeven_batch_on_cheapest_silicon": be.get("breakeven_batch", ""),
                "breakeven_extrapolated": be.get("extrapolated", ""),
                "breakeven_gpu_rate_at_b32": be.get("breakeven_gpu_rate_at_b32", ""),
                "read_at": s["read_at"],
            })
        if prices:
            summary.append({
                "model": model, "n_sellers": len(prices), "mix_prompt": P, "mix_output": O, "u": u, "batch": batch,
                "seller_min": round(min(prices), 4), "seller_median": round(st.median(prices), 4), "seller_max": round(max(prices), 4),
                "seller_max_over_min": round(max(prices) / min(prices), 2),
                "floor_min_silicon": cheapest["silicon"], "floor_min": cheapest[ukey],
                "floor_max_silicon": dearest["silicon"], "floor_max": dearest[ukey],
                "floor_max_over_min": round(dearest[ukey] / cheapest[ukey], 2),
                "median_margin_vs_cheapest_floor": round(1 - cheapest[ukey] / st.median(prices), 3),
                "H1_any_seller_below_floor_u90": any(r["below_floor_u90"] for r in margin_rows if r["model"] == model),
                "H2_price_spread_exceeds_floor_spread": max(prices) / min(prices) > dearest[ukey] / cheapest[ukey],
                "H3_median_margin_over_50pct": (1 - cheapest[ukey] / st.median(prices)) > 0.5,
            })
    _write(out / "margins.csv", margin_rows)
    _write(out / "summary.csv", summary)

    print(f"tokenprice read_at {read_at}; u={u}; batch={batch}; headline mix P~2048 O=256\n")
    for s in summary:
        print(f"{s['model']}: {s['n_sellers']} sellers  price ${s['seller_min']:.3f} / ${s['seller_median']:.3f} / ${s['seller_max']:.3f} per M (min/med/max)"
              f"  floor ${s['floor_min']:.3f} ({s['floor_min_silicon']}) to ${s['floor_max']:.3f} ({s['floor_max_silicon']})")
        print(f"   price spread {s['seller_max_over_min']}x vs floor spread {s['floor_max_over_min']}x; median margin vs cheapest floor "
              f"{s['median_margin_vs_cheapest_floor']*100:.0f}%  H1 below-floor:{s['H1_any_seller_below_floor_u90']}  "
              f"H2:{s['H2_price_spread_exceeds_floor_spread']}  H3:{s['H3_median_margin_over_50pct']}")
    print()
    for r in margin_rows:
        print(f"  {r['model']:<14} {r['provider']:<12} {r['quantization'] or '-':<5} ${r['seller_usd_per_m_at_mix']:.3f}/M  "
              f"margin {r['implied_margin_vs_cheapest']*100:5.0f}% vs {r['floor_cheapest_silicon']}  "
              f"{r['implied_margin_vs_dearest']*100:5.0f}% vs {r['floor_dearest_silicon']}"
              + (f"   breakeven: batch {r['breakeven_batch_on_cheapest_silicon']}"
                 f"{' (extrapolated)' if r['breakeven_extrapolated'] else ''} or ${r['breakeven_gpu_rate_at_b32']}/hr at B=32"
                 if r["breakeven_batch_on_cheapest_silicon"] != "" else ""))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    main(a.u, a.batch)
