// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <chrono>
#include <utils.h>
#include <cstdint>
#include <cstdlib>
#include <ios>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>

#include "cached_io.h"
#include "pq_flash_index.h"
#include "aux_utils.h"

#define READ_SECTOR_LEN (size_t) 4096
#define READ_SECTOR_OFFSET(node_id)                            \
  ((_u64) node_id / nnodes_per_sector + 1) * READ_SECTOR_LEN + \
      ((_u64) node_id % nnodes_per_sector) * max_node_len
#define NEW_READ_SECTOR_OFFSET(node_id)                            \
  ((_u64) node_id / new_nnodes_per_sector + 1) * READ_SECTOR_LEN + \
      ((_u64) node_id % new_nnodes_per_sector) * new_max_node_len
#define INF (unsigned) 0xffffffff

void convert_data_sq(std::string index_file, std::string max_min_file,
                     uint32_t thread_num) {
  /* auto data_reader = std::ifstream(data_file); */
  /* data_reader.read(reinterpret_cast<char*>(&npts), 4); */
  /* data_reader.read(reinterpret_cast<char*>(&dims), 4); */

  auto    index_reader = std::ifstream(index_file, std::ios_base::binary);
  int32_t meta_nums = 0, meta_dim = 0;
  index_reader.read((char*) (&meta_nums), sizeof(int32_t));
  index_reader.read((char*) (&meta_dim), sizeof(int32_t));
  std::cout << "index_file " << index_file << "have meta nums " << meta_nums
            << " meta_dim " << meta_dim << std::endl;

  std::unique_ptr<uint64_t[]> metas = std::make_unique<uint64_t[]>(meta_nums);
  index_reader.read((char*) metas.get(), meta_nums * sizeof(uint64_t));

  uint64_t npts, dims, max_node_len, nnodes_per_sector;
  npts = metas[0], dims = metas[1], max_node_len = metas[3],
  nnodes_per_sector = metas[4];
  std::cout << "index metas npts " << npts << " dims " << dims
            << " max_node_len " << max_node_len << " nnodes_per_sector "
            << nnodes_per_sector << " index_size " << metas[8] << std::endl;

  uint64_t round_down_dims = (dims / 8) * 8;
  uint64_t round_up_dims = std::ceil((float) dims / 8) * 8;
  std::cout << "round down dims " << round_down_dims << " round_up_dims "
            << round_up_dims << std::endl;
  float* maxs =
      (float*) aligned_alloc(32, sizeof(float) * round_up_dims * thread_num);
  float* final_maxs = (float*) aligned_alloc(32, sizeof(float) * round_up_dims);
  float* mins =
      (float*) aligned_alloc(32, sizeof(float) * round_up_dims * thread_num);
  float* final_mins = (float*) aligned_alloc(32, sizeof(float) * round_up_dims);
  for (uint64_t i = 0; i < round_up_dims * thread_num; i++) {
    maxs[i] = std::numeric_limits<float>::min();
    mins[i] = std::numeric_limits<float>::max();
  }
  for (uint64_t i = 0; i < dims; ++i) {
    final_maxs[i] = std::numeric_limits<float>::min();
    final_mins[i] = std::numeric_limits<float>::max();
  }

  uint64_t total_sector_nums = std::ceil((double) npts / nnodes_per_sector);
  uint64_t index_size = SECTOR_LEN + SECTOR_LEN * total_sector_nums;
  std::cout << "index size " << index_size << std::endl;
  std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(index_size);
  index_reader.seekg(0, std::ios_base::beg);
  index_reader.read((char*) mem_index.get(), index_size);

  auto sq_train = [&](uint32_t i) {
    uint64_t start_id = (npts / thread_num) * i;
    uint64_t end_id = std::min((npts / thread_num) * (i + 1), npts);
    float*   min_t = mins + (size_t) i * round_up_dims;
    float*   max_t = maxs + (size_t) i * round_up_dims;
    for (uint64_t id = start_id; id < end_id; id++) {
      float* node_data = (float*) (mem_index.get() + READ_SECTOR_OFFSET(id));
      for (uint64_t j = 0; j < round_down_dims; j += 8) {
        __m256 min_vec = _mm256_load_ps(min_t + j);
        __m256 max_vec = _mm256_load_ps(max_t + j);
        __m256 vec = _mm256_loadu_ps(node_data + j);
        max_vec = _mm256_max_ps(max_vec, vec);
        min_vec = _mm256_min_ps(min_vec, vec);
        _mm256_store_ps(min_t + j, min_vec);
        _mm256_store_ps(max_t + j, max_vec);
      }
      for (uint64_t j = round_down_dims; j < dims; j++) {
        min_t[j] = std::min(node_data[j], min_t[j]);
        max_t[j] = std::max(node_data[j], max_t[j]);
      }
    }
  };
  std::vector<std::thread> worker(thread_num);

  for (uint32_t i = 0; i < thread_num; i++) {
    worker[i] = std::thread(sq_train, i);
  }

  // reduction
  for (uint32_t i = 0; i < thread_num; i++) {
    worker[i].join();
    for (uint32_t j = 0; j < dims; ++j) {
      final_maxs[j] = std::max(maxs[i * round_up_dims + j], final_maxs[j]);
      final_mins[j] = std::min(mins[i * round_up_dims + j], final_mins[j]);
    }
  }

  uint64_t new_max_node_len = max_node_len - dims * sizeof(float) + dims;
  uint64_t new_nnodes_per_sector = SECTOR_LEN / new_max_node_len;
  uint64_t new_sector_nums =
      (uint64_t) std::ceil((double) npts / new_nnodes_per_sector) + 1;
  uint64_t new_file_size = new_sector_nums * SECTOR_LEN;

  std::unique_ptr<char[]> new_mem_index =
      std::make_unique<char[]>(new_file_size);
  memcpy(new_mem_index.get(), mem_index.get(), SECTOR_LEN);
  uint64_t* new_metas = (uint64_t*) (new_mem_index.get() + 8);
  new_metas[3] = new_max_node_len;
  new_metas[4] = new_nnodes_per_sector;
  new_metas[8] = new_file_size;
  std::cout << "new index metas: new_max_node_len " << new_max_node_len
            << " new_nnodes_per_sector " << new_nnodes_per_sector
            << " new file size " << new_file_size << std::endl;

  float* frac = (float*) aligned_alloc(32, sizeof(float) * round_up_dims);
  for (uint32_t i = 0; i < dims; ++i) {
    frac[i] =
        (final_maxs[i] - final_mins[i]) / std::numeric_limits<uint8_t>::max();
  }
  const int    imm8 = 0xd8;
  const size_t node_vec_size = sizeof(float) * dims;
  const size_t adj_size = max_node_len - node_vec_size;
#pragma omp parallel for num_threads(thread_num)
  for (uint32_t i = 0; i < npts; ++i) {
    char*  mem_index_node = mem_index.get() + READ_SECTOR_OFFSET(i);
    float* node_data = (float*) mem_index_node;
    char*  new_mem_index_node =
        (new_mem_index.get() + NEW_READ_SECTOR_OFFSET(i));
    uint8_t* new_node_data = (uint8_t*) (new_mem_index_node);
    for (uint32_t j = 0; j < round_down_dims; j += 8) {
      __m256 frac_vec = _mm256_load_ps(frac + j);
      __m256 min_vec = _mm256_load_ps(final_mins + j);
      __m256 data_vec = _mm256_loadu_ps(node_data + j);
      data_vec = _mm256_sub_ps(data_vec, min_vec);
      data_vec = _mm256_div_ps(data_vec, frac_vec);
      // will round to nearest int
      __m256i data_vec_i = _mm256_cvtps_epi32(data_vec);
      data_vec_i = _mm256_packs_epi32(data_vec_i, data_vec_i);
      // see intel simd instruction, using for shuffle
      data_vec_i = _mm256_permute4x64_epi64(data_vec_i, imm8);
      data_vec_i = _mm256_packus_epi16(data_vec_i, data_vec_i);
      _mm_storel_epi64((__m128i*) (new_node_data + j),
                       _mm256_castsi256_si128(data_vec_i));
    }
    for (uint32_t j = round_down_dims; j < dims; ++j) {
      float tmp = node_data[j];
      tmp = (tmp - final_mins[j]) / frac[j];
      new_node_data[j] = (uint8_t) tmp;
    }
    memcpy(new_mem_index_node + dims, mem_index_node + node_vec_size, adj_size);
  }
  index_reader.close();
  auto index_writer =
      std::ofstream(index_file, std::ios_base::binary | std::ios_base::trunc);
  index_writer.write((char*) new_mem_index.get(), new_file_size);
  index_writer.close();

  uint32_t       max_min_dims = dims * 2;
  const uint32_t one = 1;
  auto           max_min_writer =
      std::ofstream(max_min_file, std::ios_base::binary | std::ios_base::trunc);
  max_min_writer.write((char*) &max_min_dims, sizeof(uint32_t));
  max_min_writer.write((char*) &one, sizeof(uint32_t));
  max_min_writer.write((char*) final_maxs, sizeof(float) * dims);
  max_min_writer.write((char*) final_mins, sizeof(float) * dims);
  std::cout << "dims " << dims << std::endl;
  std::cout << "max min file size "
            << sizeof(uint32_t) * 2 + sizeof(float) * dims * 2 << std::endl;
  max_min_writer.close();
  std::cout << "process done" << std::endl;
  free(maxs);
  free(mins);
  free(final_maxs);
  free(final_mins);
  free(frac);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "erro usage, you need give index prefix" << std::endl;
  }
  uint32_t    threads_num = omp_get_num_procs();
  std::string index_prefix(argv[1]);
  std::string index_file = index_prefix + "_disk.index";
  std::string max_min_file = index_prefix + "_sq_max_min.bin";
  if (argc == 3) {
    threads_num = atoi(argv[2]);
  }
  std::cout << "using index file " << index_file << " save max_min file in "
            << max_min_file << " use thread nums " << threads_num << std::endl;
  convert_data_sq(index_file, max_min_file, threads_num);
  return 0;
}
