#!/bin/sh

float_dataset() {
  BASE_PATH=../tests_data/rand_float_10D_10K_norm1.0.bin
  QUERY_FILE=../tests_data/rand_float_10D_10K_norm1.0.bin
  GT_FILE=../tests_data/l2_rand_float_10D_10K_norm1.0_self_gt10
  PREFIX=float_10k
  DATA_TYPE=float
  DIST_FN=l2
  B=0.00003
  K=5
  DATA_DIM=10
  DATA_N=10000
}

uint8_dataset() {
  BASE_PATH=../tests_data/rand_uint8_10D_10K_norm50.0.bin
  QUERY_FILE=../tests_data/rand_uint8_10D_10K_norm50.0.bin
  GT_FILE=../tests_data/l2_rand_uint8_10D_10K_norm50.0_self_gt10
  PREFIX=int_10k
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.00003
  K=5
  DATA_DIM=10
  DATA_N=10000
}

# DATASET_PLACEHOLDER

##################
#   Disk Build   #
##################
R=16
BUILD_L=32
M=1
BUILD_T=16

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=16
MEM_BUILD_L=32
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.001
MEM_USE_FREQ=0
MEM_FREQ_USE_RATE=0.001

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
GP_TIMES=5
GP_T=16
GP_USE_FREQ=0
GP_LOCK_NUMS=0
GP_CUT=4096 

##############
#   Search   #
##############
BM_LIST=(2)
T_LIST=(16)
CACHE=0
MEM_L=0 # non-zero to enable
MEM_TOPK=3

#############
#    SQ     #  
#############
USE_SQ=0


# Page Search
USE_PAGE_SEARCH=0 # Set 0 for beam search, 1 for page search (default)
PS_USE_RATIO=1.0

# KNN
LS="10 12 14 16"

# Range search
RS_LS="80 100"
