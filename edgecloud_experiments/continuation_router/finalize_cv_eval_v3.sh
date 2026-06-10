#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
cd "$ROOT"

OUT_ROOT=build/continuation_router/cv_eval_val_unseen_v3
REPORT=edgecloud_experiments/reports/CONTINUATION_VERIFIED_ROUTER_RESULTS_V3_20260512_ZH.md

echo "[watch] waiting for v3 continuation-router eval processes"
while pgrep -af "eval_cv_edgecloud_r2r.py.*cv_eval_val_unseen_v3" >/dev/null; do
  sleep 120
done

if [[ ! -d "$OUT_ROOT" ]]; then
  echo "No v3 eval directory found: $OUT_ROOT" >&2
  exit 0
fi

"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("build/continuation_router/cv_eval_val_unseen_v3")
rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(d.glob("*.jsonl"))
    eps = []
    for f in files:
        with open(f) as fh:
            eps.extend(json.loads(line) for line in fh if line.strip())
    if not eps:
        continue
    rec = {
        "run": d.name,
        "n": len(eps),
        "sr": 100 * float(np.mean([float(x.get("success", 0.0)) for x in eps])),
        "spl": 100 * float(np.mean([float(x.get("spl", 0.0)) for x in eps])),
        "nav_error": float(np.mean([float(x.get("nav_error", 0.0)) for x in eps])),
        "cloud": 100 * float(np.mean([float(x.get("cloud_call_rate", 0.0)) for x in eps])),
        "cloud_calls_per_ep": float(np.mean([float(x.get("cloud_calls", 0.0)) for x in eps])),
        "steps_per_ep": float(np.mean([float(x.get("total_steps", 0.0)) for x in eps])),
    }
    (d / "result_summary.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    rows.append(rec)

report = Path("edgecloud_experiments/reports/CONTINUATION_VERIFIED_ROUTER_RESULTS_V3_20260512_ZH.md")
report.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "# Continuation-Verified Router v3 Routed-State Aggregation Results",
    "",
    "该文件由 `finalize_cv_eval_v3.sh` 自动生成。",
    "",
    "| Run | N | SR | SPL | NavErr | Cloud | Calls/Ep | Steps/Ep |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['run']} | {r['n']} | {r['sr']:.2f} | {r['spl']:.2f} | "
        f"{r['nav_error']:.2f} | {r['cloud']:.2f} | {r['cloud_calls_per_ep']:.2f} | {r['steps_per_ep']:.2f} |"
    )
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY

echo "[done] wrote $REPORT"
