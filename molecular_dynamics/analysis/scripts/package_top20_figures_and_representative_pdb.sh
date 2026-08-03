#!/usr/bin/env bash
set -euo pipefail
cd /home/jnz/AIGT_esmc/Gromacs
mkdir -p md_analysis_all/reports/top20_figures
mkdir -p md_analysis_all/reports/representative_pdb

while read -r cand; do
  mkdir -p "md_analysis_all/reports/top20_figures/$cand"
  cp md_analysis_all/candidates/$cand/*.png "md_analysis_all/reports/top20_figures/$cand/" 2>/dev/null || true
  cp md_analysis_all/candidates/$cand/extra/*.png "md_analysis_all/reports/top20_figures/$cand/" 2>/dev/null || true
done < md_analysis_all/tables/top20_candidates.txt

cp md_analysis_all/candidates/rank024_GWHPBECS013086/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank024_GWHPBECS013086.pdb
cp md_analysis_all/candidates/rank072_GWHPBECS019993/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank072_GWHPBECS019993.pdb
cp md_analysis_all/candidates/rank134_GWHPBECS005379/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank134_GWHPBECS005379.pdb
cp md_analysis_all/candidates/rank094_GWHPBECS025961/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank094_GWHPBECS025961.pdb
cp md_analysis_all/candidates/rank018_GWHPBECS034039/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank018_GWHPBECS034039.pdb
cp md_analysis_all/candidates/rank074_GWHPBECS039786/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank074_GWHPBECS039786.pdb
cp md_analysis_all/candidates/rank100_GWHPBECS025958/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank100_GWHPBECS025958.pdb
cp md_analysis_all/candidates/rank001_GWHPBECS001198/final_protein_ligand.pdb md_analysis_all/reports/representative_pdb/rank001_GWHPBECS001198.pdb

tar -czf md_analysis_all/reports/top20_figures_and_representative_pdb.tar.gz -C md_analysis_all/reports top20_figures representative_pdb
echo "Created: /home/jnz/AIGT_esmc/Gromacs/md_analysis_all/reports/top20_figures_and_representative_pdb.tar.gz"
