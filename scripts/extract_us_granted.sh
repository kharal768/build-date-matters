#!/usr/bin/env bash
# Build the robotics (CPC B25J) extracts used by src/us_release_comparison.py and
# src/us_release_gain_sources.py from two PatentsView granted-patent releases.
#
# Inputs (in the folder given as the first argument):
#   g_cpc_current_2025-03.tsv.zip   release whose g_cpc_current.tsv is dated 2025-03-13
#   g_cpc_current_2026-04.tsv.zip   release of 2026-04-10
#   g_application_2026-04.tsv.zip   release of 2026-04-10
# Output: <in_dir>/vintage/b25j_pairs_<release>.tsv and <in_dir>/vintage/b25j_filing.tsv
set -euo pipefail
IN="${1:?usage: extract_us_granted.sh <folder with the PatentsView zips>}"
cd "$IN"; mkdir -p vintage
for v in 2025-03 2026-04; do
  unzip -p "g_cpc_current_$v.tsv.zip" | awk -F'\t' '$5=="\"B25J\""{gsub(/"/,"",$1);print $1}' | sort -u > "ids_$v"
  unzip -p "g_cpc_current_$v.tsv.zip" | awk -F'\t' 'NR==FNR{a[$1];next}{gsub(/"/,"",$1);gsub(/"/,"",$5)} ($1 in a){print $1"\t"$5}' "ids_$v" - | sort -u > "vintage/b25j_pairs_$v.tsv"
done
cat ids_2025-03 ids_2026-04 | sort -u > ids
unzip -p g_application_2026-04.tsv.zip | awk -F'\t' 'NR==FNR{a[$1];next}{gsub(/"/,"")} ($2 in a){print $2"\t"$4}' ids - > vintage/b25j_filing.tsv
rm -f ids ids_2025-03 ids_2026-04
