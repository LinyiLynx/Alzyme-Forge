#!/usr/bin/env bash
set -u

INV="md_analysis_all/tables/file_inventory.csv"
SCRIPT="md_analysis_all/scripts/01_analyze_one_mdanalysis.py"
STATUS="md_analysis_all/tables/batch_analysis_status.csv"
SIM_NS="2.0"

echo "candidate,status" > "$STATUS"

tail -n +2 "$INV" | while IFS=, read -r cand src dcd fp log cg pp lp tpr trr edr st
do
    if [ "$st" != "OK" ]; then
        echo "$cand,SKIP_$st" >> "$STATUS"
        continue
    fi

    top="$src/work/solv_ions.gro"
    traj="$src/outputs/prod.dcd"
    out="md_analysis_all/candidates/$cand"
    lg="md_analysis_all/logs/$cand.analysis.log"

    if [ ! -f "$top" ]; then
        echo "$cand,MISSING_SOLV_IONS_GRO" >> "$STATUS"
        continue
    fi

    if [ ! -f "$traj" ]; then
        echo "$cand,MISSING_PROD_DCD" >> "$STATUS"
        continue
    fi

    if [ -s "$out/summary.csv" ]; then
        echo "$cand,EXISTS" >> "$STATUS"
        continue
    fi

    echo "[RUN] $cand"

    python "$SCRIPT" \
        --candidate "$cand" \
        --topology "$top" \
        --trajectory "$traj" \
        --outdir "$out" \
        --sim-ns "$SIM_NS" \
        --ligand-resname LIG \
        > "$lg" 2>&1

    if [ $? -eq 0 ]; then
        echo "$cand,OK" >> "$STATUS"
    else
        echo "$cand,FAIL" >> "$STATUS"
    fi
done

echo "Done. See $STATUS"