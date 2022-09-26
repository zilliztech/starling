# SIFT
# BASE_PATH=/data/datasets/sift10M/base10M.fbin
# QUERY_FILE=/data/datasets/sift10M/query.fbin
# GT_FILE=/data/datasets/sift10M/10M-topk1000-gt
# PREFIX=sift_10m
# DATA_TYPE=float
# DIST_FN=l2
# B=0.3
# K=10
DATA_DIM=128
DATA_N=10000000

# SSNPP
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