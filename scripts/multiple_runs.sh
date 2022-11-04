#!/bin/bash

SUMMARY_PATH=../indices/summary.log

GREEN='\033[0;32m'
NC='\033[0m'

rm $SUMMARY_PATH

for l in 5 25 50 ; do
  echo "RS_LS=\"${l}\"" >> config_local.sh

for bm in 4 8 16; do
  echo "BM_LIST=(${bm})" >> config_local.sh
# for ml in 50 100 200; do
# for ml in 50 200; do
#   echo "MEM_L=${ml}" >> config_local.sh

# for knn_ml in 20 40 80; do
#   echo "RS_KNN_MEM_L=${ml}" >> config_local.sh

# for ks in 500; do
  echo "KICKED_SIZE=${ks}" >> config_local.sh
  for i in $(seq 1 $1); do
    printf "${GREEN}Run $i ${NC}\n"
    ./run_benchmark.sh release search range
  done
done
done
# done
# done

printf "${GREEN}Summary${NC}\n"
cat $SUMMARY_PATH | grep -E "([0-9]+(\.[0-9]+\s+)){5,}"
