#include "pq_flash_index.h"

namespace diskann {
  template<typename T>
  void PQFlashIndex<T>::generate_node_nbrs_freq(
        const std::string& freq_save_path,
        const size_t query_num,
        const T *query, const size_t query_aligned_dim, const _u64 k_search,
        const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data, QueryStats *stats, const _u32 mem_L) {

    this->count_visited_nodes = true;
    this->count_visited_nbrs = true;

    init_node_visit_counter();

    nbrs_freq_counter_.resize(this->num_points);
    for (auto& m : nbrs_freq_counter_) m.clear();

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      cached_beam_search(
          query + (i * query_aligned_dim), k_search, l_search,
          res_ids + (i * k_search),
          res_dists + (i * k_search),
          beam_width, io_limit, use_reorder_data, stats + i, mem_L);
    }
    this->count_visited_nbrs = false;
    this->count_visited_nodes = false;

    // save freq file
    const std::string freq_file = freq_save_path + "_freq.bin";
    std::ofstream writer(freq_file, std::ios::binary | std::ios::out);
    diskann::cout << "Writing visited nodes and neighbors frequency: " << freq_file << std::endl;
    unsigned num = node_visit_counter.size(); // number of data points
    if (num != this->num_points) {
      diskann::cerr << "Total number of elements mismatch" << std::endl;
      exit(1);
    }
    writer.write((char *)&num, sizeof(unsigned));

    for (size_t i = 0; i < num; ++i) {
      writer.write((char *)(&(node_visit_counter[i].second)), sizeof(unsigned));
    }
    for (size_t i = 0; i < num; ++i) {
      unsigned p_size = nbrs_freq_counter_[i].size();
      writer.write((char *)(&p_size), sizeof(unsigned));
      for (const auto [nbr_id, nbr_freq] : nbrs_freq_counter_[i]) {
        writer.write((char *)(&nbr_id), sizeof(unsigned));
        writer.write((char *)(&nbr_freq), sizeof(unsigned));
      }
    }
    diskann::cout << "Writing frequency file finished" << std::endl;
  }

  // instantiations
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;
} // namespace diskann