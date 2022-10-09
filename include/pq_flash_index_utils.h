#pragma once

#include "distance.h"
#include "cosine_similarity.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present with in the graph part
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % nnodes_per_sector) * max_node_len)

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) \
  (((_u64)(id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) \
  ((((_u64)(id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace diskann {
  namespace pq_flash_index_utils {
    inline void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                          const _u8 *all_coords, const _u64 ndims, _u8 *out) {
      for (_u64 i = 0; i < n_ids; i++) {
        memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
      }
    }

    inline void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                        const _u64 pq_nchunks, const float *pq_dists,
                        float *dists_out) {
      _mm_prefetch((char *) dists_out, _MM_HINT_T0);
      _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
      memset(dists_out, 0, n_pts * sizeof(float));
      for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float *chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
          _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
          _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
          dists_out[idx] += chunk_dists[pq_centerid];
        }
      }
    }
  }
}  // namespace