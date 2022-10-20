#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncomment the line below
# If multiple lines are uncommented, only the last dataset is effective
dataset_sift10M
#dataset_ssnpp10M
#dataset_bigann10M

##################
#   Disk Build   #
##################
R=48
BUILD_L=128
M=32
BUILD_T=64

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=48
MEM_BUILD_L=128
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
MEM_USE_FREQ=1
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
GP_LOCK_NUMS=0 #lock nodes at init, this nodes will not do partition
GP_USE_FREQ=0


##############
#   Search   #
##############
BM_LIST=(4)
T_LIST=(16)
CACHE=0
MEM_L=0 # non-zero to enable
MEM_TOPK=10

# Page Search
USE_PAGE_SEARCH=1 # Set 0 for beam search, 1 for page search (default)
PS_USE_RATIO=1.0

# KNN
LS="100 120"

# Range search
RS_LS="80 100"