//
// Created by Songlin Wu on 2022/6/30.
//
#include <chrono>
#include <string>
#include <utils.h>
#include <memory>
#include <set>
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
#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * READ_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

const std::string partition_index_filename = "_tmp.index";

// Write DiskANN sector data according to graph-partition layout 
// The new index data
void relayout(const char* indexname, const char* partition_name) {
  _u64                               C;
  _u64                               _partition_nums;
  _u64                               _nd;
  _u64                               max_node_len;
  std::vector<std::vector<unsigned>> layout;
  std::vector<std::vector<unsigned>> _partition;

  std::ifstream part(partition_name);
  part.read((char*) &C, sizeof(_u64));
  part.read((char*) &_partition_nums, sizeof(_u64));
  part.read((char*) &_nd, sizeof(_u64));
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;
  
  auto meta_pair = diskann::get_disk_index_meta(indexname);
  _u64 actual_index_size = get_file_size(indexname);
  _u64 expected_file_size, expected_npts;

  if (meta_pair.first) {
      // new version
      expected_file_size = meta_pair.second.back();
      expected_npts = meta_pair.second.front();
  } else {
      expected_file_size = meta_pair.second.front();
      expected_npts = meta_pair.second[1];
  }

  if (expected_file_size != actual_index_size) {
    diskann::cout << "File size mismatch for " << indexname
                  << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size << std::endl;
    exit(-1);
  }
  if (expected_npts != _nd) {
    diskann::cout << "expect _nd: " << _nd
                  << " actual _nd: " << expected_npts << std::endl;
    exit(-1);
  }
  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];
  if (SECTOR_LEN / max_node_len != C) {
    diskann::cout << "nnodes per sector: " << SECTOR_LEN / max_node_len << " C: " << C
                  << std::endl;
    exit(-1);
  }

  layout.resize(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; i++) {
    unsigned s;
    part.read((char*) &s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*) layout[i].data(), sizeof(unsigned) * s);
  }
  part.close();

  _u64            read_blk_size = 64 * 1024 * 1024;
  _u64            write_blk_size = read_blk_size;

  std::string partition_path(partition_name);
  partition_path = partition_path.substr(0, partition_path.find_last_of('.')) + partition_index_filename;
  cached_ofstream diskann_writer(partition_path, write_blk_size);
  // cached_ifstream diskann_reader(indexname, read_blk_size);

  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
  std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);

  // this time, we load all index into mem;
  std::cout << "nnodes per sector "<<nnodes_per_sector << std::endl;
  _u64 file_size = READ_SECTOR_LEN + READ_SECTOR_LEN * (_nd / nnodes_per_sector);
  std::unique_ptr<char[]> mem_index =
      std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(indexname);
  diskann_reader.read(mem_index.get(),file_size);
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;
  *((_u64*)mem_index.get() + 4) = C;
  *((_u64*)mem_index.get()) = _partition_nums * SECTOR_LEN + SECTOR_LEN;
  std::cout << "size "<<_partition_nums *SECTOR_LEN + SECTOR_LEN << std::endl;
  diskann_writer.write((char*) mem_index.get(),
                       SECTOR_LEN);  // copy meta data;
  for (unsigned i = 0; i < _partition_nums; i++) {
    if (i % 10000 == 0) {
      diskann::cout << "relayout has done " << (float) i / _partition_nums
                    << std::endl;
      diskann::cout.flush();
    }
    memset(sector_buf.get(), 0, SECTOR_LEN);
    for (unsigned j = 0; j < layout[i].size(); j++) {
      unsigned id = layout[i][j];
      memset(node_buf.get(), 0, max_node_len);
      uint64_t index_offset = READ_SECTOR_OFFSET(id);
      uint64_t buf_offset = (uint64_t)j * max_node_len;
      memcpy((char*) sector_buf.get() + buf_offset,
             (char*) mem_index.get() + index_offset, max_node_len);
    }
    // memcpy((char*)sector_buf.get() + C*max_node_len, (char*)layout[i].data(), sizeof(unsigned));
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);
  }
  diskann::cout << "Relayout index." << std::endl;
}

int main(int argc, char** argv){
  char* indexName = argv[1];
  char* partitonName = argv[2];
  relayout(indexName, partitonName);
  return 0;
}