#!/usr/bin/env bash
set -u

TOP="md_analysis_all/tables/top20_candidates.txt"
INV="md_analysis_all/tables/file_inventory.csv"
SCRIPT="md_analysis_all/scripts/04_extra_analysis_one.py"
STATUS="md_analysis_all/tables/extra_top20_status.csv"

SIM_NS="2.0"
LIG="LIG"

echo "candidate,status" > "$STATUS"

while read -r cand
do
    if [ -z "$cand" ]; then
        continue
    fi

    src=$(awk -F, -v c="$cand" '$1==c {print $2}' "$INV")

    if [ -z "$src" ]; then
        echo "$cand,MISSING_SOURCE_DIR" >> "$STATUS"
        continue
    fi

    top="$src/work/solv_ions.gro"
    traj="$src/outputs/prod.dcd"
    out="md_analysis_all/candidates/$cand/extra"
    log="md_analysis_all/logs/$cand.extra.log"

    if [ ! -f "$top" ]; then
        echo "$cand,MISSING_TOPOLOGY" >> "$STATUS"
        continue
    fi

    if [ ! -f "$traj" ]; then
        echo "$cand,MISSING_TRAJECTORY" >> "$STATUS"
        continue
    fi

    echo "[RUN] $cand"

    python "$SCRIPT" \
      --candidate "$cand" \
      --topology "$top" \
      --trajectory "$traj" \
      --outdir "$out" \
      --sim-ns "$SIM_NS" \
      --ligand-resname "$LIG" \
      > "$log" 2>&1

    if [ $? -eq 0 ]; then
        echo "$cand,OK" >> "$STATUS"
    else
        echo "$cand,FAIL" >> "$STATUS"
    fi

done < "$TOP"

echo "Done. See $STATUS"