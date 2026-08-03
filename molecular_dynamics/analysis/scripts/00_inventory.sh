#!/usr/bin/env bash
set -euo pipefail

LIST="md_analysis_all/tables/candidate_dirs.txt"
OUT="md_analysis_all/tables/file_inventory.csv"

echo "candidate,source_dir,dcd,final_pdb,prod_log,complex_gro,protein_pdb,ligand_pdb,em_tpr,em_trr,em_edr,status" > "$OUT"

while read -r d; do
    candidate=$(basename "$d")

    dcd=$(find "$d" -type f -path "*/outputs/prod.dcd" | head -n 1 || true)
    final_pdb=$(find "$d" -type f -path "*/outputs/final.pdb" | head -n 1 || true)
    prod_log=$(find "$d" -type f -path "*/outputs/prod.log" | head -n 1 || true)
    complex_gro=$(find "$d" -type f -path "*/work/complex.gro" | head -n 1 || true)
    protein_pdb=$(find "$d" -type f -path "*/work/protein.pdb" | head -n 1 || true)
    ligand_pdb=$(find "$d" -type f -path "*/work/ligand.pdb" | head -n 1 || true)
    em_tpr=$(find "$d" -type f -path "*/work/em.tpr" | head -n 1 || true)
    em_trr=$(find "$d" -type f -path "*/work/em.trr" | head -n 1 || true)
    em_edr=$(find "$d" -type f -path "*/work/em.edr" | head -n 1 || true)

    missing=""

    [ -z "$dcd" ] && missing="${missing};MISSING_DCD"
    [ -z "$final_pdb" ] && missing="${missing};MISSING_FINAL_PDB"
    [ -z "$prod_log" ] && missing="${missing};MISSING_PROD_LOG"
    [ -z "$complex_gro" ] && missing="${missing};MISSING_COMPLEX_GRO"
    [ -z "$protein_pdb" ] && missing="${missing};MISSING_PROTEIN_PDB"
    [ -z "$ligand_pdb" ] && missing="${missing};MISSING_LIGAND_PDB"

    if [ -z "$missing" ]; then
        status="OK"
    else
        status="${missing#;}"
    fi

    echo "$candidate,$d,$dcd,$final_pdb,$prod_log,$complex_gro,$protein_pdb,$ligand_pdb,$em_tpr,$em_trr,$em_edr,$status" >> "$OUT"
done < "$LIST"

echo "Wrote $OUT"
