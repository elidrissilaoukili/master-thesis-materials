"""
pruning_tables.py
-----------------
Reads the 7 pruning JSON files and generates pruning_tables.html
containing Table 1 (summary) and Table 2 (full comparison).

Usage:
    python pruning_tables.py
    python pruning_tables.py --input-dir ./data --output pruning_report.html

Dependencies: standard library only (json, pathlib, argparse, html)
"""

import json
import html as html_mod
import argparse
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def pct(v):
    """Format a 0-1 float as a percentage string."""
    return f"{v * 100:.2f}%"


def drop_class(drop: float) -> str:
    """Return CSS class for accuracy drop colouring."""
    if drop < 0:
        return "d-gain"
    elif drop <= 0.005:
        return "d-ok"
    elif drop <= 0.02:
        return "d-warn"
    else:
        return "d-bad"


def fmt_drop(drop: float) -> str:
    """Format accuracy drop with sign."""
    if drop < 0:
        return f"−{abs(drop) * 100:.2f}%"
    elif drop == 0:
        return "0.00%"
    else:
        return f"+{drop * 100:.2f}%"


def fmt_int(v: int) -> str:
    return f"{v:,}"


def e(s) -> str:
    """HTML-escape a value."""
    return html_mod.escape(str(s))


# ── loaders ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Table 1: Summary ─────────────────────────────────────────────────────────

SUMMARY_ROWS = [
    {
        "num": "1",
        "method": "Unstructured",
        "granularity": ("b-gray", "weight"),
        "criterion": "L1 global magnitude",
        "best_sparsity": "70%",
        "accuracy": "92.76%",
        "drop": -0.0006,
        "active_params": 7_093_444,
        "sparse_mb": 27.06,
        "complexity": ("b-blue", "Low"),
        "note": "No GPU speedup; disk unchanged",
    },
    {
        "num": "2",
        "method": "Structured",
        "granularity": ("b-amber", "filter"),
        "criterion": "L1 filter norm",
        "best_sparsity": "10%",
        "accuracy": "93.22%",
        "drop": -0.0002,
        "active_params": 21_200_808,
        "sparse_mb": 80.87,
        "complexity": ("b-blue", "Low"),
        "note": "Real speedup after rebuild; aggressive ratios hurt",
    },
    {
        "num": "3",
        "method": "Magnitude",
        "granularity": ("b-gray", "weight"),
        "criterion": "Global L1",
        "best_sparsity": "30%",
        "accuracy": "93.21%",
        "drop": -0.0001,
        "active_params": 16_480_528,
        "sparse_mb": 62.87,
        "complexity": ("b-blue", "Low"),
        "note": "Global L1 beats local methods at high sparsity",
    },
    {
        "num": "4",
        "method": "Movement",
        "granularity": ("b-gray", "weight"),
        "criterion": "Taylor |w·∇|",
        "best_sparsity": "30%",
        "accuracy": "92.80%",
        "drop": 0.004,
        "active_params": 15_989_963,
        "sparse_mb": 61.00,
        "complexity": ("b-amber", "Medium"),
        "note": "Data-aware; collapses above 70% sparsity",
    },
    {
        "num": "5",
        "method": "Lottery Ticket",
        "granularity": ("b-gray", "weight"),
        "criterion": "Iterative mask",
        "best_sparsity": "30% target / 6.9% actual",
        "accuracy": "93.21%",
        "drop": -0.0001,
        "active_params": 21_905_088,
        "sparse_mb": 83.56,
        "complexity": ("b-red", "High"),
        "note": "Ticket mask preserves acc; random init → 10%",
    },
    {
        "num": "6",
        "method": "Iterative",
        "granularity": ("b-gray", "weight"),
        "criterion": "Global L1, 15%/round",
        "best_sparsity": "62.3%",
        "accuracy": "93.22%",
        "drop": -0.0002,
        "active_params": 8_903_967,
        "sparse_mb": 33.97,
        "complexity": ("b-amber", "Medium"),
        "note": "Best sparsity/accuracy ratio; cliff after round 12",
    },
    {
        "num": "7",
        "method": "One-shot",
        "granularity": ("b-gray", "weight"),
        "criterion": "L1 global, single pass",
        "best_sparsity": "30%",
        "accuracy": "93.21%",
        "drop": -0.0001,
        "active_params": 16_480_528,
        "sparse_mb": 62.87,
        "complexity": ("b-blue", "Low"),
        "note": "Fastest; random pruning catastrophic (~10% acc)",
    },
]


def build_summary_table() -> str:
    header = """
    <thead>
      <tr>
        <th>#</th><th>Method</th><th>Granularity</th><th>Criterion</th>
        <th class="r">Best Sparsity</th><th class="r">Accuracy</th>
        <th class="r">Acc. Drop</th><th class="r">Active Params</th>
        <th class="r">Sparse MB</th><th>Complexity</th><th>Key Note</th>
      </tr>
    </thead>"""

    rows = []
    for r in SUMMARY_ROWS:
        gran_cls, gran_lbl = r["granularity"]
        cplx_cls, cplx_lbl = r["complexity"]
        dc = drop_class(r["drop"])
        rows.append(f"""
      <tr>
        <td class="muted">{e(r['num'])}</td>
        <td class="name">{e(r['method'])}</td>
        <td><span class="badge {gran_cls}">{e(gran_lbl)}</span></td>
        <td>{e(r['criterion'])}</td>
        <td class="r">{e(r['best_sparsity'])}</td>
        <td class="r">{e(r['accuracy'])}</td>
        <td class="r {dc}">{fmt_drop(r['drop'])}</td>
        <td class="r">{fmt_int(r['active_params'])}</td>
        <td class="r">{r['sparse_mb']:.2f}</td>
        <td><span class="badge {cplx_cls}">{e(cplx_lbl)}</span></td>
        <td class="muted">{e(r['note'])}</td>
      </tr>""")

    return f"""
  <table>
    {header}
    <tbody>{''.join(rows)}
    </tbody>
  </table>"""


# ── Table 2: Full comparison data builders ───────────────────────────────────

def tr(method, variant, target, actual, acc, drop, f1, params, sparse_mb, gpu_ms, cpu_ms):
    dc = drop_class(drop)
    tgt_str = f"{target * 100:.0f}%" if isinstance(target, float) else str(target)
    return f"""
      <tr>
        <td class="muted">{e(method)}</td>
        <td>{e(variant)}</td>
        <td class="r">{tgt_str}</td>
        <td class="r">{actual * 100:.1f}%</td>
        <td class="r">{acc * 100:.2f}%</td>
        <td class="r {dc}">{fmt_drop(drop)}</td>
        <td class="r">{f1:.4f}</td>
        <td class="r">{fmt_int(params)}</td>
        <td class="r">{sparse_mb:.2f}</td>
        <td class="r">{gpu_ms:.1f}</td>
        <td class="r">{cpu_ms:.0f}</td>
      </tr>"""


def gh(num, label):
    return f"""
      <tr class="gh"><td colspan="11"><span class="gh-num">{num} ·</span> {e(label)}</td></tr>"""


def build_full_table(data: dict) -> str:
    d1 = data["unstructured"]
    d2 = data["structured"]
    d3 = data["magnitude"]
    d4 = data["movement"]
    d5 = data["lth"]
    d6 = data["iterative"]
    d7 = data["oneshot"]

    header = """
    <thead>
      <tr>
        <th>Method</th><th>Variant / Criterion</th>
        <th class="r">Target Sparsity</th><th class="r">Actual Sparsity</th>
        <th class="r">Accuracy</th><th class="r">Acc. Drop</th><th class="r">F1</th>
        <th class="r">Active Params</th><th class="r">Sparse MB</th>
        <th class="r">GPU ms</th><th class="r">CPU ms</th>
      </tr>
    </thead>"""

    rows = []

    # 1 — Unstructured
    rows.append(gh("1", "Unstructured — L1 Global Magnitude"))
    for r in d1["results"]:
        rows.append(tr(
            "Unstructured", "L1 global",
            r["target_sparsity"], r["actual_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    # 2 — Structured
    rows.append(gh("2", "Structured — L1 Filter Norm"))
    for r in d2["results"]:
        rows.append(tr(
            "Structured", "L1 filter",
            r["filter_pruning_ratio"], r["structured_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    # 3 — Magnitude (group by sparsity, show all 3 criteria)
    rows.append(gh("3", "Magnitude — Local L1 / Local L2 / Global L1"))
    for r in d3["results"]:
        rows.append(tr(
            "Magnitude", r["criterion"],
            r["target_sparsity"], r["actual_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    # 4 — Movement
    rows.append(gh("4", "Movement — Taylor |w·∇| (Data-Aware, 10 calibration batches)"))
    for r in d4["results"]:
        rows.append(tr(
            "Movement", "Taylor |w·∇|",
            r["target_sparsity"], r["actual_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    # 5 — LTH
    rows.append(gh("5", "Lottery Ticket — Iterative Mask, 5 Rounds (Frankle & Carlin 2019)"))
    for r in d5["results"]:
        tgt_str = f"{int(r['target_sparsity'] * 100)}% →"
        rows.append(f"""
      <tr>
        <td class="muted">LTH</td>
        <td>Trained weights</td>
        <td class="r">{tgt_str}</td>
        <td class="r">{r['actual_sparsity'] * 100:.1f}%</td>
        <td class="r">{r['winning_ticket_accuracy'] * 100:.2f}%</td>
        <td class="r {drop_class(r['winning_ticket_accuracy_drop'])}">{fmt_drop(r['winning_ticket_accuracy_drop'])}</td>
        <td class="r">{r['winning_ticket_f1']:.4f}</td>
        <td class="r">{fmt_int(r['params_active'])}</td>
        <td class="r">{r['size_sparse_theoretical_mb']:.2f}</td>
        <td class="r">{r['inference_gpu_ms']:.1f}</td>
        <td class="r">{r['inference_cpu_ms']:.0f}</td>
      </tr>""")

    # 6 — Iterative (subset of key rounds)
    rows.append(gh("6", "Iterative — Global L1, 15% per Round (15 Rounds Total)"))
    key_rounds = {1, 4, 6, 10, 12, 15}
    labels = {6: "Round 6 ★ best"}
    for r in d6["trajectory"]:
        rnd = r["round"]
        if rnd not in key_rounds:
            continue
        lbl = labels.get(rnd, f"Round {rnd}")
        rows.append(tr(
            "Iterative", lbl,
            "—", r["cumulative_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    # 7 — One-shot (l1 global + random; skip l2 for brevity)
    rows.append(gh("7", "One-shot — L1 Global &amp; Random Control"))
    for r in d7["results"]:
        if r["variant"] == "oneshot_l2_global":
            continue
        variant_label = "L1 global" if r["variant"] == "oneshot_l1_global" else "Random"
        rows.append(tr(
            "One-shot", variant_label,
            r["target_sparsity"], r["actual_sparsity"],
            r["accuracy"], r["accuracy_drop"],
            r["f1"], r["params_active"],
            r["size_sparse_theoretical_mb"],
            r["inference_gpu_ms"], r["inference_cpu_ms"],
        ))

    return f"""
  <table>
    {header}
    <tbody>{''.join(rows)}
    </tbody>
  </table>"""


# ── HTML skeleton ─────────────────────────────────────────────────────────────

CSS = """
  :root {
    --bg:      #f7f5f0;
    --surface: #ffffff;
    --surface2:#f2efe8;
    --border:  #e0dbd0;
    --border2: #ccc7b8;
    --text:    #1a1714;
    --muted:   #7a7468;
    --accent:  #1a5c3a;
    --gain:    #0f5fa8;
    --good:    #1a6e3a;
    --warn:    #9a5c10;
    --danger:  #b02020;
    --mono: 'IBM Plex Mono', monospace;
    --serif:'Fraunces', serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    background:var(--bg); color:var(--text);
    font-family:var(--mono); font-size:12.5px; line-height:1.55;
    padding:52px 40px 80px; min-width:960px;
  }
  .page-header {
    margin-bottom:52px;
    display:grid; grid-template-columns:1fr auto; align-items:end; gap:24px;
    padding-bottom:20px; border-bottom:2px solid var(--text);
  }
  .page-header h1 {
    font-family:var(--serif); font-size:30px; font-weight:700;
    letter-spacing:-0.03em; line-height:1.1;
  }
  .page-header h1 em { font-style:italic; font-weight:300; color:var(--muted); }
  .meta-block { text-align:right; font-size:11px; color:var(--muted); line-height:1.8; }
  .meta-block strong { color:var(--text); font-weight:600; }
  .section { margin-bottom:64px; }
  .section-title { display:flex; align-items:baseline; gap:12px; margin-bottom:14px; }
  .section-title .num {
    font-family:var(--serif); font-size:11px; font-weight:600;
    letter-spacing:0.06em; color:var(--muted); text-transform:uppercase;
    border:1px solid var(--border2); padding:1px 8px; border-radius:2px;
  }
  .section-title h2 { font-family:var(--serif); font-size:16px; font-weight:600; letter-spacing:-0.01em; }
  .section-title .desc { font-size:11px; color:var(--muted); margin-left:auto; font-style:italic; }
  .tbl-wrap {
    border:1px solid var(--border2); border-radius:4px; overflow-x:auto;
    background:var(--surface); box-shadow:0 1px 4px rgba(0,0,0,.06);
  }
  table { width:100%; border-collapse:collapse; }
  thead tr { background:var(--surface2); border-bottom:1.5px solid var(--border2); }
  th {
    font-family:var(--mono); font-size:10px; font-weight:600;
    letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
    padding:10px 14px; text-align:left; white-space:nowrap;
  }
  th.r { text-align:right; }
  tbody tr { border-bottom:1px solid var(--border); transition:background .1s; }
  tbody tr:last-child { border-bottom:none; }
  tbody tr:hover { background:#faf8f3; }
  td { padding:10px 14px; font-family:var(--mono); font-size:11.5px; white-space:nowrap; vertical-align:middle; }
  td.r  { text-align:right; font-variant-numeric:tabular-nums; }
  td.muted { color:var(--muted); font-size:11px; }
  td.name { font-family:var(--serif); font-size:13px; font-weight:600; letter-spacing:-0.01em; }
  .badge {
    display:inline-block; font-family:var(--mono); font-size:9.5px;
    font-weight:600; letter-spacing:0.05em; padding:2px 7px;
    border-radius:2px; text-transform:uppercase;
  }
  .b-green { background:#e8f4ed; color:#1a5c3a; border:1px solid #b6d9c3; }
  .b-blue  { background:#e8eef7; color:#2d5fa8; border:1px solid #b0c4e4; }
  .b-amber { background:#faf0e0; color:#9a5c10; border:1px solid #e0c090; }
  .b-red   { background:#faeaea; color:#b02020; border:1px solid #e0b0b0; }
  .b-gray  { background:#f0ede6; color:#6a6458; border:1px solid #ccc7b8; }
  .d-gain  { color:var(--gain);   font-weight:500; }
  .d-ok    { color:var(--good);   font-weight:500; }
  .d-warn  { color:var(--warn);   font-weight:500; }
  .d-bad   { color:var(--danger); font-weight:500; }
  .d-neut  { color:var(--muted); }
  tr.gh td {
    background:#f5f2ea; color:var(--muted);
    font-family:var(--serif); font-size:10.5px; font-weight:600;
    letter-spacing:0.07em; text-transform:uppercase;
    padding:5px 14px; border-top:1.5px solid var(--border2); border-bottom:1px solid var(--border);
  }
  tr.gh td .gh-num { color:var(--accent); margin-right:6px; font-family:var(--mono); }
  .foot { margin-top:10px; font-size:10.5px; color:var(--muted); line-height:1.7; }
  .foot .c-gain { color:var(--gain); }
  .foot .c-ok   { color:var(--good); }
  .foot .c-warn { color:var(--warn); }
  .foot .c-bad  { color:var(--danger); }
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Network Pruning — Comparison Tables</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>

<div class="page-header">
  <div>
    <h1>Neural Network Pruning<br><em>Benchmark Report</em></h1>
  </div>
  <div class="meta-block">
    <strong>Model</strong> ResNet · 23.5M params<br>
    <strong>Baseline Acc.</strong> 93.2% &nbsp;|&nbsp; <strong>Disk</strong> 90.03 MB<br>
    <strong>Device</strong> CUDA &nbsp;|&nbsp; <strong>Threshold</strong> Δacc ≤ 2%
  </div>
</div>

<!-- TABLE 1: SUMMARY -->
<div class="section">
  <div class="section-title">
    <span class="num">Table 01</span>
    <h2>Summary — Best Configuration per Method</h2>
    <span class="desc">One row per pruning method</span>
  </div>
  <div class="tbl-wrap">
    {table1}
  </div>
  <p class="foot">
    Acc. Drop = Baseline − Pruned. <span class="c-gain">■ Negative = accuracy improved.</span>
    Sparse MB = active_params × 4B (theoretical lower bound). Disk constant at 90.03 MB.
  </p>
</div>

<!-- TABLE 2: FULL COMPARISON -->
<div class="section">
  <div class="section-title">
    <span class="num">Table 02</span>
    <h2>Full Comparison — All Sparsity Levels</h2>
    <span class="desc">Every method × every sparsity point</span>
  </div>
  <div class="tbl-wrap">
    {table2}
  </div>
  <p class="foot">
    Acc. Drop color coding —
    <span class="c-gain">■ negative = improved</span> &nbsp;
    <span class="c-ok">■ 0–0.5% = negligible</span> &nbsp;
    <span class="c-warn">■ 0.5–2% = acceptable</span> &nbsp;
    <span class="c-bad">■ &gt;2% = exceeds threshold</span><br>
    LTH "Target →" denotes input target; actual sparsity reflects the 5-round iterative mask result.
    All disk sizes constant at 90.03 MB (PyTorch .pth zip).
  </p>
</div>

</body>
</html>
"""


# ── main ─────────────────────────────────────────────────────────────────────

FILE_MAP = {
    "unstructured": "1_unstructured_Pruning.json",
    "structured":   "2_structured_Pruning.json",
    "magnitude":    "3_magnitude_Pruning.json",
    "movement":     "4_movement_Pruning.json",
    "lth":          "5_lottery_ticket_Pruning.json",
    "iterative":    "6_iterative_Pruning.json",
    "oneshot":      "7_oneshot_Pruning.json",
}


def main():
    parser = argparse.ArgumentParser(description="Generate pruning comparison tables as HTML.")
    parser.add_argument("--input-dir", default=".", help="Directory containing the JSON files (default: current dir)")
    parser.add_argument("--output", default="pruning_tables.html", help="Output HTML file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load all JSON files
    data = {}
    for key, filename in FILE_MAP.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        data[key] = load_json(path)
        print(f"  ✓  Loaded {filename}")

    # Build tables
    table1_html = build_summary_table()
    table2_html = build_full_table(data)

    # Render final HTML
    output_html = HTML_TEMPLATE.format(
        css=CSS,
        table1=table1_html,
        table2=table2_html,
    )

    output_path = Path(args.output)
    output_path.write_text(output_html, encoding="utf-8")
    print(f"\n  ✓  Report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
