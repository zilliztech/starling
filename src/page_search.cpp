#include "logger.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "timer.h"

namespace {
} // namespace anonymous

namespace diskann {
  template<typename T>
  void PQFlashIndex<T>::load_partition_data(const std::string &index_prefix,
      const _u64 nnodes_per_sector, const _u64 num_points) {
    std::string partition_file = index_prefix + "_partition.bin";
    std::ifstream part(partition_file);
    _u64          C, partition_nums, nd;
    part.read((char *) &C, sizeof(_u64));
    part.read((char *) &partition_nums, sizeof(_u64));
    part.read((char *) &nd, sizeof(_u64));
    if (nnodes_per_sector && num_points &&
        (C != nnodes_per_sector || nd != num_points)) {
      diskann::cerr << "partition information not correct." << std::endl;
      exit(-1);
    }
    diskann::cout << "Partition meta: C: " << C << " partition_nums: " << partition_nums
              << " nd: " << nd << std::endl;
    this->gp_layout_.resize(partition_nums);
    for (unsigned i = 0; i < partition_nums; i++) {
      unsigned s;
      part.read((char *) &s, sizeof(unsigned));
      this->gp_layout_[i].resize(s);
      part.read((char *) gp_layout_[i].data(), sizeof(unsigned) * s);
    }
    this->id2page_.resize(nd);
    part.read((char *) id2page_.data(), sizeof(unsigned) * nd);
    diskann::cout << "Load partition data done." << std::endl;
  }

  template<typename T>
  void PQFlashIndex<T>::page_search(
      const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, float use_ratio, QueryStats *stats) {
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float        query_norm = 0;
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    for (uint32_t i = 0; i < this->data_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    // if inner product, we laso normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    IOContext &ctx = data.ctx;
    auto       query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
      float cur_expanded_dist = dist_cmp_float->compare(
          query_float, centroid_data + aligned_dim * cur_m,
          (unsigned) aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L_) {
      std::vector<unsigned> mem_results(mem_topk_);
      mem_index_->search(query, mem_topk_, mem_L_, mem_results.data()); // returns <hops, cmps>
      compute_and_add_to_retset(mem_results.data(), mem_topk_);
    } else {
      compute_and_add_to_retset(&best_medoid, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    std::vector<unsigned> last_io_ids;
    last_io_ids.reserve(2 * beam_width);
    std::vector<char> last_pages(SECTOR_LEN * beam_width * 2);
    int n_ops;

    while (k < cur_list_size && num_ios < io_limit) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;

      // distribute cache and disk-read nodes
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        unsigned pid = retset[marker].id;
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          // TODO: add different cache strategies
          auto iter = nhood_cache.find(retset[marker].id);
          if (iter != nhood_cache.end()) {
            // cached_nhoods.push_back(
            //     std::make_pair(retset[marker].id, iter->second));
            // if (stats != nullptr) {
            //   stats->n_cache_hits++;
            // }
          } else {
            frontier.push_back(retset[marker].id);
            page_visited.insert(pid);
          }
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
        }
        marker++;
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              (static_cast<_u64>(id2page_[id]+1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        n_ops = reader->submit_reqs(frontier_read_reqs, ctx);
      }

      // compute remaining nodes in the pages that are fetched in the previous round
      for (size_t i = 0; i < last_io_ids.size(); ++i) {
        const unsigned last_io_id = last_io_ids[i];
        char    *sector_buf = last_pages.data() + i * SECTOR_LEN;
        const unsigned pid = id2page_[last_io_id];
        const size_t p_size = gp_layout_[pid].size();
        // minus one for the vector that is already computed previously
        unsigned vis_size = use_ratio * (p_size - 1);
        std::vector<std::tuple<float, unsigned, char*>> vis_cand;
        vis_cand.reserve(p_size);

        // compute exact distances of the vectors within the page
        for (size_t j = 0; j < p_size; ++j) {
          const unsigned id = gp_layout_[pid][j];
          if (id == last_io_id) continue;

          char* node_buf = sector_buf + j * max_node_len;
          T *node_fp_coords_copy = data_buf;
          memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
          float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                (unsigned) aligned_dim);
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          vis_cand.push_back({cur_expanded_dist, id, sector_buf + j * max_node_len});
        }
        if (vis_size && vis_size != p_size) {
          std::sort(vis_cand.begin(), vis_cand.end());
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (size_t j = 0; j < vis_size; ++j) {
          const auto [dist, id, node_buf] = vis_cand[j];
          unsigned *nbors = OFFSET_TO_NODE_NHOOD(node_buf);
          unsigned adj_size = *(nbors++);
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < adj_size; ++m) {
            if (visited.find(nbors[m]) == visited.end()) {
              nbors[nbors_cand_size++] = nbors[m];
              visited.insert(nbors[m]);
            }
          }
          if (nbors_cand_size) {
            compute_dists(nbors, nbors_cand_size, dist_scratch);
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const int nbor_id = nbors[m];
              const float nbor_dist = dist_scratch[m];
              if (stats != nullptr) {
                stats->n_cmps++;
              }
              if (nbor_dist >= retset[cur_list_size - 1].distance &&
                  (cur_list_size == l_search))
                continue;
              Neighbor nn(nbor_id, nbor_dist, true);
              auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
              if (cur_list_size < l_search) ++cur_list_size;
              if (r < nk) nk = r;
            }
          }
        }
      }
      last_io_ids.clear();

      // process cached nhoods
      // TODO: support cache
      /*
      for (auto &cached_nhood : cached_nhoods) {
        auto  global_cache_iter = coord_cache.find(cached_nhood.first);
        T *   node_fp_coords_copy = global_cache_iter->second;
        float cur_expanded_dist;
        if (!use_disk_index_pq) {
          cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                (unsigned) aligned_dim);
        } else {
          if (metric == diskann::Metric::INNER_PRODUCT)
            cur_expanded_dist = disk_pq_table.inner_product(
                query_float, (_u8 *) node_fp_coords_copy);
          else
            cur_expanded_dist = disk_pq_table.l2_distance(
                query_float, (_u8 *) node_fp_coords_copy);
        }
        full_retset.push_back(
            Neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));

        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            visited.insert(id);
            float dist = dist_scratch[m];
            if (dist >= retset[cur_list_size - 1].distance &&
                (cur_list_size == l_search))
              continue;
            Neighbor nn(id, dist, true);
            // Return position in sorted list where nn inserted.
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < nk)
              // nk logs the best position in the retset that was
              // updated due to neighbors of n.
              nk = r;
          }
        }
      }
      */

      // get last submitted io results, blocking
      if (!frontier.empty()) {
        reader->get_events(ctx, n_ops);
      }

      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &frontier_nhood : frontier_nhoods) {
        char *sector_buf = frontier_nhood.second;
        unsigned pid = id2page_[frontier_nhood.first];
        memcpy(last_pages.data() + last_io_ids.size() * SECTOR_LEN, sector_buf, SECTOR_LEN);
        last_io_ids.emplace_back(frontier_nhood.first);

        for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
          unsigned id = gp_layout_[pid][j];
          if (id == frontier_nhood.first) {
            char *node_buf = sector_buf + j * max_node_len;
            unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
            unsigned nnbrs = *(node_nbrs++);
            T *node_fp_coords_copy = data_buf;
            memcpy(node_fp_coords_copy, node_buf, disk_bytes_per_point);
            float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy,
                                                (unsigned) aligned_dim);
            full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

            compute_dists(node_nbrs, nnbrs, dist_scratch);
            for (_u64 m = 0; m < nnbrs; ++m) {
              unsigned id = node_nbrs[m];
              if (visited.find(id) == visited.end()) {
                visited.insert(id);
                float dist = dist_scratch[m];
                if (stats != nullptr) {
                  stats->n_cmps++;
                }
                if (dist >= retset[cur_list_size - 1].distance &&
                    (cur_list_size == l_search))
                  continue;
                Neighbor nn(id, dist, true);
                auto     r = InsertIntoPool(
                    retset.data(), cur_list_size,
                    nn);  // Return position in sorted list where nn inserted.
                if (cur_list_size < l_search)
                  ++cur_list_size;
                if (r < nk)
                  nk = r;  // nk logs the best position in the retset that was
                           // updated due to neighbors of n.
              }
            }
          }
        }
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // copy k_search values
    for (_u64 i = 0, t = 0; i < full_retset.size() && t < k_search; i++) {
      if(i > 0 && full_retset[i].id == full_retset[i-1].id){
        continue;
      }
      indices[t] = full_retset[i].id;
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // flip the sign to convert min to max
          distances[t] = (-distances[t]);
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[t] *= (max_base_norm * query_norm);
        }
      }
      t++;
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;

} // namespace diskann