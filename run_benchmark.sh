#!/bin/bash

set -e
# set -x

BASE_PATH=/data/datasets/SSNPP/FB_ssnpp_database.10M.u8bin
QUERY_FILE=/data/datasets/SSNPP/FB_ssnpp_public_queries.u8bin
GT_FILE=/data/datasets/SSNPP/ssnpp-10M-gt
PREFIX=ssnpp_10m
DATA_TYPE=uint8
B=0.6
# BASE_PATH=/data/datasets/sift10M/base10M.fbin
# QUERY_FILE=/data/datasets/sift10M/query.fbin
# GT_FILE=/data/datasets/sift10M/10M-topk1000-gt
# PREFIX=sift_10m
# DATA_TYPE=float
# B=0.3

METRIC=L2
R=48
BUILD_L=128
M=32
BUILD_T=64

BM_LIST=(1 4 8 16)
T_LIST=(16)
DIST_FN=l2
CACHE=0
K=10
LS="20 30 40 50 60 70 80 90 100 110 120 130 140 150"

RADIUS=96237
RS_LS="100 150 200 250 300 350 400"


case $1 in
    debug)
        cmake -DCMAKE_BUILD_TYPE=Debug -B debug
        pushd debug
    ;;
    release)
        cmake -DCMAKE_BUILD_TYPE=Release -B release
        pushd release
    ;;
    *)
        echo "./run_benchmark.sh [debug/release] [build/search]"
    ;;
esac

make -j

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"

date

cd tests
case $2 in
    build)
        if [ -d "$INDEX_PREFIX_PATH" ]
        then
            echo "Directory $INDEX_PREFIX_PATH is already exit. Remove or rename it to rebuild."
            exit 1
        else
            mkdir -p ${INDEX_PREFIX_PATH}
        fi
        time ./build_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FN --data_path $BASE_PATH --index_path_prefix $INDEX_PREFIX_PATH -R $R -L $BUILD_L -B $B -M $M -T $BUILD_T > ${INDEX_PREFIX_PATH}build.log
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
                    SEARCH_LOG=${INDEX_PREFIX_PATH}search_K${K}_CACHE${CACHE}_BW${BW}_T${T}.log
                    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ./search_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FN --index_path_prefix $INDEX_PREFIX_PATH --query_file $QUERY_FILE --gt_file $GT_FILE -K $K --result_path ${INDEX_PREFIX_PATH}result --num_nodes_to_cache $CACHE -T $T -L ${LS} -W $BW > ${SEARCH_LOG}
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
                    SEARCH_LOG=${INDEX_PREFIX_PATH}search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}.log
                    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ./range_search_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FN --index_path_prefix $INDEX_PREFIX_PATH --num_nodes_to_cache $CACHE -T $T -W $BW --query_file $QUERY_FILE --gt_file $GT_FILE --range_threshold $RADIUS result_output_prefix ${INDEX_PREFIX_PATH}result -L $RS_LS > ${SEARCH_LOG} 
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
                echo "./run_benchmark.sh [debug/release] [build/search] [knn/range]"
                ;;
        esac
        ;;
    *)
        echo "./run_benchmark.sh [debug/release] [build/search] [knn/range]"
        ;;
esac

popd
