#!/bin/bash
source config_local.sh

INDEX_PREFIX_PATH="${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_MEM_USE_FREQ${MEM_USE_FREQ}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_GP_USE_FREQ${GP_USE_FREQ}_CUT${GP_CUT}/"
FREQ_PATH="${INDEX_PREFIX_PATH}FREQ/NQ_${FREQ_QUERY_CNT}_BM_${FREQ_BM}_L_${FREQ_L}_T_${FREQ_T}/"

print_usage_and_exit(){
    echo "Usage: ./unset.sh [compile/index_file/gp/mem_index/freq/sample_file/index_dir/relayout] [release/debug]"
    exit -1;
}
cd ../indices
case $1 in 
    compile)
        echo "remove all compiled file."
        rm -rf ../debug ../release
    ;;
    index_file)
        echo "copy the un-gp disk index to disk index"
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
        INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
        if [ -f $OLD_INDEX_FILE ]; then
            cp $OLD_INDEX_FILE $INDEX_FILE
        else 
            echo "Wrong! make sure you have the old index file copy."
        fi
    ;;
    gp)
        echo "remove gp dir and reset the gp index to no-gp index."
        echo "unset index_file."
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
        INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
        if [ -f $OLD_INDEX_FILE ]; then
            echo "copy the un-gp disk index to disk index..."
            cp $OLD_INDEX_FILE $INDEX_FILE
        fi
        echo ""
        rm -rf $GP_PATH
        rm -f ${INDEX_PREFIX_PATH}_partition.bin
    ;;
    mem_index)
        echo "remove mem index dir."
        rm -rf ${MEM_INDEX_PATH}
    ;;
    freq)
        echo "remove freq dir."
        rm -rf ${FREQ_PATH}
    ;;
    sample_file)
        echo "remove sample data dir."
        rm -rf ${MEM_SAMPLE_PATH}
    ;;
    index_dir)
        echo "remove index file dir."
        rm -rf ${INDEX_PREFIX_PATH}
    ;;
    relayout)
        case $2 in
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
    
        echo "will relayout the index file using the gpfile in gp dir."
        echo "unset index_file."
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
        INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
        if [ ! -f $INDEX_FILE ]; then
            echo "ERRO! no disk index file!"
            exit 1; 
        fi

        if [ ! -f $OLD_INDEX_FILE ]; then
            echo "no old file, will copy the index to old index file."
            cp $INDEX_FILE $OLD_INDEX_FILE
        fi

        if [ ! -d ${GP_PATH} ]; then
            echo "ERRO! no gp dir, maybe you should run './run_benchmark.sh release gp knn' first."
            exit 1;
        fi

        if [ ! -f ${GP_PATH}_part.bin ]; then
            echo "ERRO! no gp file in gp dir, maybe you should run './run_benchmark.sh release gp knn' first."
            exit 1;
        fi
        echo ${EXE_PATH}
        time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_PATH}_part.bin > ${GP_PATH}relayout.log
        cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
        cp ${GP_PATH}_part.bin ${INDEX_PREFIX_PATH}_partition.bin
    ;;
    search)
        echo rm ${INDEX_PREFIX_PATH}search
        rm ${INDEX_PREFIX_PATH}search/*
    ;;
esac
