#!/bin/bash

set -e
# set -x

source config.sh

# Disk Index Build
R=48
BUILD_L=128
M=32
BUILD_T=64

# In-Memory Navigation Graph BUILD
MEM_R=48
MEM_BUILD_L=128
MEM_ALPHA=1.2
SAMPLING_RATE=0.01

# Search
BM_LIST=(4)
T_LIST=(16)
CACHE=0
MEM_L=0
MEM_TOPK=10

# KNN
LS="100 120"

# Range search
RS_LS="80 100"


case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ../debug
    pushd ../debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release
    pushd ../release
  ;;
  *)
    echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/search] [knn/range]"
    exit
  ;;
esac
make -j

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE/"
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM/"

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]
  then
    echo "Directory $dir is already exit. Remove or rename it to rebuild."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

cd tests
date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    echo "Building disk index..."
    time ./build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      -R $R \
      -L $BUILD_L \
      -B $B \
      -M $M \
      -T $BUILD_T > ${INDEX_PREFIX_PATH}build.log
  ;;
  build_mem)
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    mkdir -p ${MEM_SAMPLE_PATH}
    # TODO: Add frequency sampling method
    echo "Generating random slice..."
    time ./utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $SAMPLING_RATE > ${MEM_SAMPLE_PATH}slice.log
    echo "Building memory index..."
    time ./build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_SAMPLE_PATH}_data.bin \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  search)
    if ! [ -d "$INDEX_PREFIX_PATH" ]
    then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi
    
    case $3 in
      knn)
        log_arr=()
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ./search_disk_index --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --mem_topk ${MEM_TOPK} > ${SEARCH_LOG}
            log_arr+=( ${SEARCH_LOG} )
          done
        done
        for f in "${log_arr[@]}"
        do
          echo $f
          cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}"
        done
      ;;
      range)
        log_arr=()
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ./range_search_disk_index \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -W $BW \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              --range_threshold $RADIUS \
              -L $RS_LS \
              --mem_L ${MEM_L} \
              --mem_topk ${MEM_TOPK} > ${SEARCH_LOG}
            log_arr+=( ${SEARCH_LOG} )
          done
        done
        for f in "${log_arr[@]}"
        do
          echo $f
          cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}"
        done
      ;;
      *)
        echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/search] [knn/range]"
      ;;
    esac
  ;;
  *)
    echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/search] [knn/range]"
  ;;
esac

popd
