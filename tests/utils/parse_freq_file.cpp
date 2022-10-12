#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include "partition_and_pq.h"
#include "utils.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>
#include <utility>
#include <vector>

template<typename T>
int aux_main(char** argv) {
  std::string base_file(argv[2]);
  std::string freq_file(argv[3]);
  std::string output_prefix(argv[4]);
  float       use_ratio = atof(argv[5]);

  std::string output_data_file = output_prefix + "_data.bin";
  std::string output_ids_file = output_prefix + "_ids.bin";

  std::ifstream freq_reader(freq_file, std::ios_base::binary);
  std::ifstream base_reader(base_file, std::ios_base::binary);
  std::ofstream data_writer(output_data_file, std::ios_base::binary);
  std::ofstream ids_writer(output_ids_file, std::ios_base::binary);

  unsigned npts = 0;
  freq_reader.read((char*) &npts, sizeof(unsigned));
  std::vector<unsigned> freq_vec(npts);
  freq_reader.read((char*) freq_vec.data(), sizeof(unsigned) * npts);

  std::vector<std::pair<unsigned, unsigned>> freq_pair_vec(npts);
  for (unsigned i = 0; i < npts; i++) {
    freq_pair_vec[i] = std::make_pair(i, freq_vec[i]);
  }
  std::sort(
      freq_pair_vec.begin(), freq_pair_vec.end(),
      [](std::pair<unsigned, unsigned>& a, std::pair<unsigned, unsigned>& b) {
        return a.second > b.second;
      });
  size_t   dim = 0, nums = 0;
  unsigned one = 1;

  diskann::get_bin_metadata(base_file, nums, dim);
  unsigned _dim = dim;
  unsigned use_npt = nums * use_ratio;
  assert(nums == npts);
  assert(use_npt > 0);
  auto buf = std::make_unique<char[]>(sizeof(T) * dim);

  data_writer.write((char*) &use_npt, sizeof(unsigned));
  data_writer.write((char*) &_dim, sizeof(unsigned));

  ids_writer.write((char*) &use_npt, sizeof(unsigned));
  ids_writer.write((char*) &one, sizeof(unsigned));

  for (unsigned i = 0; i < use_npt; i++) {
    unsigned node_id = freq_pair_vec[i].first;
    ids_writer.write((char*) &node_id, sizeof(unsigned));
    unsigned offset = 8 + sizeof(T) * dim * node_id;
    base_reader.seekg(offset, std::ios_base::beg);
    base_reader.read((char*) buf.get(), sizeof(T) * dim);
    data_writer.write((char*) buf.get(), sizeof(T) * dim);
  }

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0]
              << " data_type [float/int8/uint8] base_bin_file "
                 "freq_file_path output_prefix sample_ratio"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float")) {
    aux_main<float>(argv);
  } else if (std::string(argv[1]) == std::string("int8")) {
    aux_main<int8_t>(argv);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    aux_main<uint8_t>(argv);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
  return 0;
}
