"""
Figure 1: seller asking prices vs measured cost floors, one panel per model.

    python figure.py   -> results/figure1.svg

Reads results/floors.csv and results/margins.csv written by margin.py. Floors are
drawn as bands from u=0.9 (left edge, cheap) to u=0.3 (right edge, dear) per silicon
at the headline cell; sellers are dots. Log x axis. No dependencies.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
W, PAD_L, PAD_R = 920, 170, 60
ROW = 22
BATCH, OUT = 32, 256
SIL_COLOR = {"mi300x": "#c2410c", "a100-80g": "#0e7490", "l40s": "#4d7c0f", "h100-pcie": "#6d28d9", "h100-sxm": "#1d4ed8"}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def main() -> None:
    floors = [r for r in csv.DictReader((ROOT / "results" / "floors.csv").open())
              if int(r["batch"]) == BATCH and int(r["output"]) == OUT and 1500 <= int(r["prompt"]) <= 2500 and r["precision"] == "bf16"]
    margins = list(csv.DictReader((ROOT / "results" / "margins.csv").open()))
    models = sorted({m["model"] for m in margins})
    # the corpus holds repeat runs per cell; collapse to one row per (model, silicon), averaging the floors
    grp = defaultdict(list)
    for f in floors:
        grp[(f["model"], f["silicon"])].append(f)
    by_floor = defaultdict(list)
    for (model, _), rows in grp.items():
        f = dict(rows[0])
        for k in ("floor_usd_per_m_u30", "floor_usd_per_m_u60", "floor_usd_per_m_u90"):
            f[k] = str(sum(float(r[k]) for r in rows) / len(rows))
        by_floor[model].append(f)

    allv = [float(f["floor_usd_per_m_u30"]) for f in floors] + [float(f["floor_usd_per_m_u90"]) for f in floors] \
        + [float(m["seller_usd_per_m_at_mix"]) for m in margins]
    lo10, hi10 = math.floor(math.log10(min(allv))), math.ceil(math.log10(max(allv)))
    X = lambda v: PAD_L + (math.log10(v) - lo10) / (hi10 - lo10) * (W - PAD_L - PAD_R)  # noqa: E731

    panels = []
    y = 70
    out = []
    for model in models:
        fl = sorted(by_floor[model], key=lambda f: float(f["floor_usd_per_m_u60"]))
        sellers = sorted([m for m in margins if m["model"] == model], key=lambda m: float(m["seller_usd_per_m_at_mix"]))
        top = y
        out.append(f'<text x="{PAD_L}" y="{y}" font-size="14" font-weight="600">{esc(model)}</text>')
        out.append(f'<text x="{W - PAD_R}" y="{y}" text-anchor="end" opacity=".6">{len(sellers)} sellers, {len(fl)} silicon type{"s" if len(fl) != 1 else ""} measured</text>')
        y += 14
        for f in fl:
            y += ROW
            x0, x1, xm = X(float(f["floor_usd_per_m_u90"])), X(float(f["floor_usd_per_m_u30"])), X(float(f["floor_usd_per_m_u60"]))
            c = SIL_COLOR.get(f["silicon"], "#555")
            out.append(f'<text x="{PAD_L - 10}" y="{y + 4}" text-anchor="end" fill="{c}">{esc(f["silicon"])} floor</text>')
            out.append(f'<rect x="{x0:.1f}" y="{y - 7}" width="{x1 - x0:.1f}" height="14" fill="{c}" fill-opacity=".18"/>')
            out.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{y - 7}" y2="{y + 7}" stroke="{c}" stroke-width="2"/>')
            out.append(f'<text x="{x1 + 6:.1f}" y="{y + 4}" font-size="10" fill="{c}" opacity=".8">${f["gpu_rate"]}/hr {f["rate_provenance"]}</text>')
        y += ROW + 4
        out.append(f'<text x="{PAD_L - 10}" y="{y + 4}" text-anchor="end">sellers</text>')
        cheapest_u90 = min(float(f["floor_usd_per_m_u90"]) for f in fl)
        placed: list[float] = []
        for m in sellers:
            v = float(m["seller_usd_per_m_at_mix"])
            below = v < cheapest_u90
            col = "#b91c1c" if below else "#1f2937"
            x = X(v)
            tier = sum(1 for px in placed if x - px < 95)   # one row down per neighbor whose label would overlap
            placed.append(x)
            out.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="{col}"/>')
            lab = f'{m["provider"]} ${v:.3f}' + (" (below every floor)" if below else "")
            ly = y + 18 + 12 * tier
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y + 6}" y2="{ly - 9}" stroke="{col}" stroke-opacity=".4"/>')
            out.append(f'<text x="{x:.1f}" y="{ly}" font-size="10" text-anchor="middle" fill="{col}">{esc(lab)}</text>')
        y += 64 + 12 * max(0, len(sellers) - 3)
        panels.append((top, y))
        y += 18

    H = y + 30
    hdr = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12" fill="#222">',
           f'<rect width="{W}" height="{H}" fill="#fff" rx="6"/>',
           f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600">Asking price vs measured cost floor, USD per M tokens at a 2048:256 mix, batch {BATCH}</text>',
           f'<text x="{PAD_L}" y="42" opacity=".65">Band = floor on rented GPU from 90% utilization (left) to 30% (right), bar at 60%. Red dot = seller priced below every floor at 90%.</text>']
    for e in range(lo10, hi10 + 1):
        x = X(10 ** e)
        hdr.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="52" y2="{H - 22}" stroke="#000" stroke-opacity=".1"/>')
        hdr.append(f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle" opacity=".6">${10 ** e:g}</text>')
    (ROOT / "results" / "figure1.svg").write_text("\n".join(hdr + out + ["</svg>"]))


if __name__ == "__main__":
    main()
