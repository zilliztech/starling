#!/bin/bash

set -e
# set -x

source config_main.sh

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_DESCEND_${GP_DESCEND_TIMES}/"
FREQ_PATH="${INDEX_PREFIX_PATH}FREQ/NQ_${FREQ_QUERY_CNT}_BM_${FREQ_BM}_L_${FREQ_L}_T_${FREQ_T}/"

print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/freq/gp/search] [knn/range]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ../debug
    EXE_PATH=../debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release
    EXE_PATH=../release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
pushd $EXE_PATH
make -j
popd

mkdir -p ../indices && cd ../indices

date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/build_disk_index \
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
    time ${EXE_PATH}/tests/utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $MEM_RAND_SAMPLING_RATE > ${MEM_SAMPLE_PATH}sample.log
    echo "Building memory index..."
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_SAMPLE_PATH}_data.bin \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  freq)
    check_dir_and_make_if_absent ${FREQ_PATH}
    FREQ_LOG="${FREQ_PATH}freq.log"

    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f $DISK_FILE_PATH ]; then
      DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    fi

    echo "Generating frequency file... ${FREQ_LOG}"
    time ${EXE_PATH}/tests/search_disk_index_save_freq \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --freq_save_path $FREQ_PATH \
              --query_file $FREQ_QUERY_FILE \
              --expected_query_num $FREQ_QUERY_CNT \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${FREQ_PATH}result \
              --num_nodes_to_cache ${FREQ_CACHE} \
              -T $FREQ_T \
              -L $FREQ_L \
              -W $FREQ_BM \
              --mem_L ${MEM_L} \
              --mem_topk ${MEM_TOPK} \
              --use_page_search 0 \
              --disk_file_path ${DISK_FILE_PATH} > ${FREQ_LOG}
  ;;
  gp)
    check_dir_and_make_if_absent ${GP_PATH}
    OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f "$OLD_INDEX_FILE" ]; then
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
    fi
    GP_FILE_PATH=${GP_PATH}_part.bin
    echo "Running graph partition... ${GP_FILE_PATH}.log"
    time ${EXE_PATH}/graph_partition/SSD_Based_Plan $DATA_DIM $DATA_N ${OLD_INDEX_FILE} \
      $DATA_TYPE $GP_FILE_PATH $GP_T $GP_TIMES $GP_DESCEND_TIMES 0 > ${GP_FILE_PATH}.log
    
    echo "Running relayout... ${GP_PATH}relayout.log"
    time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} > ${GP_PATH}relayout.log
    if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
      mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
    fi
    #TODO: Use only one index file
    cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
    cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
  ;;
  search)
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $USE_PAGE_SEARCH -eq 1 ]; then
      if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
        echo "Partition file not found. Run the script with gp option first."
        exit 1
      fi
      echo "Using Page Search"
    else
      if [ -f ${DISK_FILE_PATH} ]; then
        DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
      fi
      echo "Using Beam Search"
    fi

    case $3 in
      knn)
        log_arr=()
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
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
              --mem_topk ${MEM_TOPK} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --disk_file_path ${DISK_FILE_PATH}> ${SEARCH_LOG}
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
        # TODO: Range search needs to be modified
        echo "Support only KNN for now"
        exit 0

        log_arr=()
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/range_search_disk_index \
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
        print_usage_and_exit
      ;;
    esac
  ;;
  *)
    print_usage_and_exit
  ;;
esac