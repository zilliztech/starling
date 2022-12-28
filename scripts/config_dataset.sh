#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

###############
#   SIFT10M   #
###############
dataset_sift10M() {
  BASE_PATH=/data/datasets/sift10M/base10M.fbin
  QUERY_FILE=/data/datasets/sift10M/query.fbin
  GT_FILE=/data/datasets/sift10M/10M-topk1000-gt
  PREFIX=sift_10m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=10000000
}

################
#   SSNPP10M   #
################
dataset_ssnpp10M() {
  BASE_PATH=/data/datasets/SSNPP/FB_ssnpp_database.10M.u8bin
  QUERY_FILE=/data/datasets/SSNPP/FB_ssnpp_public_queries.u8bin
  GT_FILE=/data/datasets/SSNPP/ssnpp-10M-gt
  PREFIX=ssnpp_10m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.6
  RADIUS=96237
  DATA_DIM=256
  DATA_N=10000000
}

dataset_ssnpp16M() {
  BASE_PATH=/data/exper_datasets/SSNPP/base.16M.u8bin
  QUERY_FILE=/data/exper_datasets/SSNPP/FB_ssnpp_public_queries.u8bin
  GT_FILE=/data/exper_datasets/SSNPP/ssnpp-1000-103929.0.bin
  # GEN_FREQ_QUERY=/data/datasets/SSNPP/FB_ssnpp_public_queries.u8bin
  PREFIX=ssnpp_16m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.96
  RADIUS=103929
  DATA_DIM=256
  DATA_N=16000000
}

#################
#   BIGANN10M   #
#################
dataset_bigann10M() {
  BASE_PATH=/data/datasets/BIGANN/base.10M.u8bin
  QUERY_FILE=/data/datasets/BIGANN/query.public.10K.128.u8bin
  GT_FILE=/data/datasets/BIGANN/bigann-10M-gt.bin 
  PREFIX=bigann_10m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=10000000
}

dataset_bigann33M() {
  BASE_PATH=/data/exper_datasets/BIGANN/base.33M.u8bin
  QUERY_FILE=/data/exper_datasets/BIGANN/query.public.10K.128.u8bin
  GT_FILE=/data/exper_datasets/BIGANN/bigann-1000-2727.0.bin
  #GEN_FREQ_QUERY=/data/datasets/BIGANN/freq_sample/train_90k.bin
  PREFIX=bigann_33m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.5
  K=10
  RADIUS=2727
  DATA_DIM=128
  DATA_N=33000000
}