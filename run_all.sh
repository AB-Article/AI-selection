#!/usr/bin/env bash
#
# Reproduce every number, table and figure in the paper and its supplement.
#
#   ./run_all.sh          full pipeline (~25 min; the last stage dominates)
#   ./run_all.sh fast     everything except the robustness simulation (~6 min)
#
# Inputs are read from data/ and all outputs are written to results/ and
# figures/.  Nothing outside those two directories is modified, so a run can
# always be checked against the archived copies by diffing them.
#
# Override the locations with DATA= and OUT= if you keep them elsewhere.

set -euo pipefail

export DATA="${DATA:-data}"
export OUT="${OUT:-results}"
FIGS="${FIGS:-figures}"
MODE="${1:-full}"

mkdir -p "$OUT" "$FIGS"

run () {
    printf '  %-26s ' "$1"
    if python3 "scripts/$1.py" > "$OUT/logs/$1.log" 2>&1; then
        echo "ok"
    else
        echo "FAILED  (see $OUT/logs/$1.log)"; exit 1
    fi
}

mkdir -p "$OUT/logs"

echo "[1/5] primary corpus: fits, identities, decision experiments"
run analysis_main          # -> results.json, table_cells.csv, table_rules.csv
run stats_meta             # -> stats_meta.json, table_meta/aicc/splithalf.csv
run floor_identification   # -> floor_identification.json, table_floor.csv
run decision_logspace      # -> decision_logspace.json

echo "[2/5] out-of-domain, language models, retired benchmarks"
run stats_extended         # -> stats_extended.json, table_extended.csv
run stats_llm              # -> stats_llm.json, table_llm.csv
run latency_ladders        # -> stats_latladders.json, table_latladders.csv
run latency_final          # -> stats_latfinal.json
run joint_llm              # -> stats_joint.json
run analysis_retired       # -> table_retired*.csv, stats_retired.json
run retired_stats          # -> stats_retired_final.json, table_censoring.csv

echo "[3/5] audits and external criterion"
run bias_audit             # -> bias_audit.json, table_bias.csv
run noise_model            # -> noise_model.json
run verify_minimax         # numerical check of Theorem 11(iii)-(iv)
run verify_endpoints       # numerical check of the reported endpoints
run reconcile              # -> reconciled.json
run finalize               # -> final_estimate.json, table_final.csv
printf '  %-26s ' "external_criterion"
python3 scripts/external_criterion.py --selftest \
        > "$OUT/logs/external_criterion.log" 2>&1 && echo "ok"

if [ "$MODE" != "fast" ]; then
    echo "[4/5] robustness simulations (slow: winner's curse, censoring, noise)"
    run robustness         # -> robustness.json
else
    echo "[4/5] robustness simulations           skipped (fast mode)"
fi

echo "[5/5] figures"
for f in make_figs_is make_figs_ext make_fig_llm make_fig_ladders \
         make_fig_retired make_figure; do
    run "$f"
done
mv -f "$OUT"/*.pdf "$OUT"/*.png "$FIGS"/ 2>/dev/null || true

echo
echo "Done.  Numbers in $OUT/, figures in $FIGS/."
echo "Headline values:"
python3 - <<'PY'
import json, os
out = os.environ.get("OUT", "results")
r = json.load(open(f"{out}/results.json"))
m = json.load(open(f"{out}/stats_meta.json"))
print(f"  pooled price          {m['pooled']:.4f}  "
      f"CI [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]")
print(f"  prediction interval   [{m['pi'][0]:.4f}, {m['pi'][1]:.4f}]")
print(f"  agreement at 0.35     {r['agreement']:.2f} %   kappa {r['kappa']:.4f}")
print(f"  median per-cell price {r['lambda_hat']:.4f}")
PY
