#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncomment the line below
# If multiple lines are uncommented, only the last dataset is effective
# dataset_bigann10M

##################
#   Disk Build   #
##################
R=48
BUILD_L=128
M=32
BUILD_T=8

##################
#       SQ       #
##################
USE_SQ=0

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=48
MEM_BUILD_L=128
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
MEM_USE_FREQ=0
MEM_FREQ_USE_RATE=0.01

##########################
#   Generate Frequency   #
##########################
FREQ_QUERY_FILE=$QUERY_FILE
FREQ_QUERY_CNT=0 # Set 0 to use all (default)
FREQ_BM=4
FREQ_L=100 # only support one value at a time for now
FREQ_T=16
FREQ_CACHE=0
FREQ_MEM_L=0 # non-zero to enable
FREQ_MEM_TOPK=10

#######################
#   Graph Partition   #
#######################
GP_TIMES=16
GP_T=16
GP_LOCK_NUMS=0 # will lock nodes at init, the lock_node_nums = partition_size * GP_LOCK_NUMS
GP_USE_FREQ=0 # use freq file to partition graph
GP_CUT=4096 # the graph's degree will been limited at 4096


##############
#   Search   #
##############
BM_LIST=(4)
T_LIST=(8)
CACHE=0
MEM_L=0 # non-zero to enable

# Page Search
USE_PAGE_SEARCH=1 # Set 0 for beam search, 1 for page search (default)
PS_USE_RATIO=1.0

# KNN
LS="100"

# Range search
RS_LS="80"
RS_ITER_KNN_TO_RANGE_SEARCH=1 # 0 for custom search, 1 for iterating via KNN, combine with USE_PAGE_SEARCH
KICKED_SIZE=0 # non-zero to reuse intermediate states during page search
RS_CUSTOM_ROUND=0 # set when use custom search, 0 for all pages within radius
