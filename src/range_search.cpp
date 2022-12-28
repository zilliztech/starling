#include "pq_flash_index.h"
#include "timer.h"

namespace diskann {
  /*
  typedef struct {
    unsigned pid_;
    bool calc_all_ = true;
    float score_ = 0;
    std::vector<Neighbor*> nodes_to_vis;
  } Page;

  class CandV {
  public:
    CandV(const _u64 L, const _u32 beam_width, 
      tsl::robin_set<_u64>& nv, tsl::robin_set<unsigned>& pv, std::unordered_map<unsigned, unsigned>& id2pid)
       : L_(L), bw_(beam_width), node_visited_(nv), page_visited_(pv), id2pid_(id2pid) {};

    size_t size() {
      return cands_.size();
    }

    void insert(Neighbor&& nn) {
      // TODO: change to binary search and insert for min_cands_ to optimize large L
      unsigned pid = id2pid_[nn.id];
      if (page_visited_.find(pid) == page_visited_.end()) {
        cands_.push_back(nn);
        if (stats_.find(pid) == stats_.end()) {
          stats_[pid].pid_ = pid;
          stats_[pid].calc_all_ = true;
          stats_[pid].score_ = nn.distance;
        } else {
          stats_[pid].score_ = std::min(stats_[pid].score_, nn.distance);
        }

        stats_[pid].nodes_to_vis.push_back(&(cands_.back()));
      }
    }

    std::vector<Page> get_best_bw_pids() {
      std::vector<std::pair<float, unsigned>> p_cands;
      for (const auto& [pid, p] : stats_) {
        p_cands.emplace_back(p.score_, p.pid_);
      }
      std::sort(p_cands.begin(), p_cands.end()); // TODO: bm-select

      std::vector<Page> res;
      for (_u32 i = 0; i < bw_; ++i) {
        unsigned pid = p_cands[i].second;
        res.push_back(std::move(stats_[pid]));
        stats_.erase(pid);
      }
      return res;
    }
    
    inline bool isDone(size_t res_cnt) {
      return vis_size_ >= L_ && (res_cnt == 0 || vis_size_ >= cands_.size());
    }
  private:
    _u64 L_;
    _u32 bw_;
    tsl::robin_set<_u64> &node_visited_;
    tsl::robin_set<unsigned> &page_visited_;
    std::unordered_map<unsigned, unsigned> &id2pid_;

    size_t vis_size_ = 0;
    // size_t back_marker = 0;
    std::vector<Neighbor> cands_;
    std::unordered_map<unsigned, Page> stats_;
  };
  */

  template<typename T>
  _u32 PQFlashIndex<T>::custom_range_search(const T *query1,
                        const double range,
                        const _u32          mem_L,
                        const _u64          knn_min_l_search,
                        const _u64          max_l_search,
                        std::vector<_u64>  &indices,
                        std::vector<float> &distances,
                        const _u32          beam_width,
                        const float         page_search_use_ratio,
                        const _u32          kicked_size,
                        const _u32          custom_round_num,
                        QueryStats *        stats) {

    Timer query_timer, cpu_timer, io_timer;

    std::vector<MemNavNeighbor> mem_cands;

    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dis(mem_L);
    std::vector<_u32> mem_internal_indices(mem_L);
    std::vector<T*> mem_res = std::vector<T*>();
    mem_index_->search_with_tags(query1, mem_L, mem_L, mem_tags.data(), mem_dis.data(), mem_internal_indices.data(), mem_res);

    for (_u32 i = 0; i < mem_L; ++i) {
      // initialization set all distance to float::max()
      if (mem_dis[i] <= range) {
        mem_cands.emplace_back(mem_internal_indices[i], mem_dis[i], mem_tags[i]);
      }
    }

    // bfs to get all results in memory
    // TODO: the internal states of `search_with_tags` might be reused during bfs
    if (mem_cands.size() == mem_L) {
      mem_index_->bfs_with_tags(query1, range, mem_cands);
    }

    if (mem_cands.size() == 0) {
        return this->custom_range_search_iter_page_search(query1, range, 
            mem_L, mem_tags, mem_dis, knn_min_l_search, max_l_search,
            indices, distances, beam_width,
            page_search_use_ratio, kicked_size, stats);
    }

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

    // if inner product, we also normalize the query and set the last coordinate
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
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    tsl::robin_set<_u64> &visited = *(query_scratch->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_scratch->page_visited);

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    // no vector is within the radius
    /*
    if (mem_cands.empty()) {
      for (_u32 i = 0; i < std::min(mem_L, mem_cand_size_when_empty); ++i) {
      // for (_u32 i = 0; i < mem_L; ++i) {
        if (mem_dis[i] >= std::numeric_limits<float>::max()) break;
        mem_cands.emplace_back(mem_internal_indices[i], mem_dis[i], mem_tags[i]);
      }

      // failed to search knn results, fallback to medoid
      if (mem_cands.empty()) {
        _u32                        best_medoid = 0;
        float                       best_dist = (std::numeric_limits<float>::max)();
        for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
          float cur_expanded_dist = dist_cmp_float->compare(
              query_float, centroid_data + aligned_dim * cur_m,
              (unsigned) aligned_dim);
        }
        diskann::cout << "Warning: fallback to medoid" << std::endl;
        mem_cands.emplace_back(0, best_dist, best_medoid);
      }
    }
    */

    std::vector<unsigned> pages_to_vis;
    std::vector<std::pair<unsigned, char*>> pages_to_io;
    std::vector<AlignedRead> page_read_reqs;
    page_read_reqs.reserve(2 * beam_width);

    for (const auto& cand : mem_cands) {
      auto pid = id2page_[cand.tag];
      visited.insert(cand.tag);
      if (page_visited.find(pid) == page_visited.end()) {
        pages_to_vis.push_back(pid);
        page_visited.insert(pid);
      }
    }

    if (stats != nullptr) {
      stats->cpu_us += cpu_timer.elapsed();
    }

    for (_u32 round = 0; ; ++round) {
      if (pages_to_vis.empty() || (custom_round_num && (round >= custom_round_num))) break;

      std::vector<unsigned> next_round_pages;
      for (size_t i = 0; i < pages_to_vis.size(); i += beam_width) {
        cpu_timer.reset();
        pages_to_io.clear();
        page_read_reqs.clear();
        for (size_t j = 0; j < beam_width && (i+j < pages_to_vis.size()); ++j) {
          std::pair<unsigned, char*> fnhood;
          fnhood.first = pages_to_vis[i+j];
          fnhood.second = sector_scratch + pages_to_io.size() * SECTOR_LEN;
          pages_to_io.push_back(fnhood);
          page_read_reqs.emplace_back(
              static_cast<_u64>((pages_to_vis[i+j]+1)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_ios++;
          }
        }
        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();
        }
        io_timer.reset();
        int n_ops = reader->submit_reqs(page_read_reqs, ctx);
        reader->get_events(ctx, n_ops);
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }

        cpu_timer.reset();
        for (const auto& fnhood : pages_to_io) {
          auto pid = fnhood.first;
          char *sector_buf = fnhood.second;
          for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
            unsigned id = gp_layout_[pid][j];
            char *node_buf = sector_buf + j * max_node_len;
            memcpy(data_buf, node_buf, disk_bytes_per_point);
            float cur_expanded_dist = dist_cmp->compare(query, data_buf,
                                                  (unsigned) aligned_dim);
            if (cur_expanded_dist <= range) {
              indices.push_back(id);
              distances.push_back(cur_expanded_dist);
            }

            unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
            unsigned nnbrs = *(node_nbrs++);
            unsigned nbors_cand_size = 0;
            for (unsigned m = 0; m < nnbrs; ++m) {
              if (visited.find(node_nbrs[m]) == visited.end()) {
                node_nbrs[nbors_cand_size++] = node_nbrs[m];
                visited.insert(node_nbrs[m]);
              }
            }
            if (nbors_cand_size) {
              compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
              for (unsigned m = 0; m < nbors_cand_size; ++m) {
                const int nbor_id = node_nbrs[m];
                const unsigned nnbr_pid = id2page_[nbor_id];
                const float nbor_dist = dist_scratch[m];
                if (nbor_dist <= range && page_visited.find(nnbr_pid) == page_visited.end()) {
                  next_round_pages.push_back(nnbr_pid);
                  page_visited.insert(nnbr_pid);
                }
              }
            }
          }
        }
        if (stats != nullptr) {
          stats->cpu_us += cpu_timer.elapsed();
        }
      }
      pages_to_vis.swap(next_round_pages);
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();

    return indices.size();
  }

  // range search returns results of all neighbors within distance of range.
  // indices and distances need to be pre-allocated of size l_search and the
  // return value is the number of matching hits.
  template<typename T>
  _u32 PQFlashIndex<T>::range_search_iter_knn(const T *query1, const double range,
                                     const _u32          mem_L,
                                     const _u64          min_l_search,
                                     const _u64          max_l_search,
                                     std::vector<_u64> & indices,
                                     std::vector<float> &distances,
                                     const _u64          min_beam_width,
                                     const float         page_search_use_ratio,
                                     QueryStats *        stats) {
    Timer query_timer;
    _u32 res_count = 0;

    bool stop_flag = false;

    _u32 l_search = min_l_search;  // starting size of the candidate list
    while (!stop_flag) {
      indices.resize(l_search);
      distances.resize(l_search);
      _u64 cur_bw =
          min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
      cur_bw = (cur_bw > 100) ? 100 : cur_bw;
      for (auto &x : distances)
        x = std::numeric_limits<float>::max();
      
      if (use_page_search_) {
        this->page_search(query1, l_search, mem_L, l_search, indices.data(),
                               distances.data(), cur_bw,
                               std::numeric_limits<_u32>::max(),
                               false, page_search_use_ratio, stats);
      } else {
        this->cached_beam_search(query1, l_search, l_search, indices.data(),
                               distances.data(), cur_bw, false, stats, mem_L);
      }
      for (_u32 i = 0; i < l_search; i++) {
        if (distances[i] > (float) range) {
          res_count = i;
          break;
        } else if (i == l_search - 1)
          res_count = l_search;
      }
      if (res_count < (_u32)(l_search / 2.0))
        stop_flag = true;
      l_search = l_search * 2;
      if (l_search > max_l_search)
        stop_flag = true;
    }
    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
  }

  template<typename T>
  _u32 PQFlashIndex<T>::custom_range_search_iter_page_search(
                                     const T *query1, const double range,
                                     const _u32          mem_L,
                                     std::vector<unsigned> &upper_mem_tags,
                                     std::vector<float> &upper_mem_dis,
                                     const _u64          min_l_search,
                                     const _u64          max_l_search,
                                     std::vector<_u64> & indices,
                                     std::vector<float> &distances,
                                     const _u64          min_beam_width,
                                     const float         page_search_use_ratio,
                                     const _u32          kicked_size,
                                     QueryStats *        stats) {
    _u32 res_count = 0;

    bool stop_flag = false;

    _u32 l_search = min_l_search;  // starting size of the candidate list
    PageSearchPersistData<T> persist_data;
    ThreadData<T>& data = persist_data.thread_data;
    data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      persist_data.thread_data = this->thread_data.pop();
    }
    persist_data.full_ret_set.reserve(4096);
    std::vector<Neighbor> &retset = persist_data.ret_set;
    NeighborVec &kicked = persist_data.kicked;
    kicked.set_cap(kicked_size);
    unsigned &cur_list_size = persist_data.cur_list_size;
    cur_list_size = 0;

    float        query_norm = 0;
    const float *query_float = data.scratch.aligned_query_float;

    for (uint32_t i = 0; i < this->data_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += query1[i] * query1[i];
    }

    if (metric == diskann::Metric::INNER_PRODUCT) {
      query_norm = std::sqrt(query_norm);
      data.scratch.aligned_query_T[this->data_dim - 1] = 0;
      data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      for (uint32_t i = 0; i < this->data_dim - 1; i++) {
        data.scratch.aligned_query_T[i] /= query_norm;
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }
    persist_data.query_norm = query_norm;

    auto       query_scratch = &(data.scratch);
    query_scratch->reset();
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);

    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      pq_flash_index_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      pq_flash_index_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
      compute_pq_dists(node_ids, n_ids, dist_scratch);
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    Timer query_timer;

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

    retset.resize(l_search+1);
    std::vector<T*> mem_res = std::vector<T*>();
    if (mem_L && upper_mem_tags.size() < mem_L) {
      upper_mem_tags.resize(mem_L);
      upper_mem_dis.resize(mem_L);
      mem_index_->search_with_tags(query1, mem_L, mem_L, upper_mem_tags.data(), upper_mem_dis.data(), nullptr, mem_res);
    }

    if (mem_L) {
      compute_and_add_to_retset(upper_mem_tags.data(), std::min(mem_L,l_search));
      // for (_u64 i = 0; i < std::min(mem_L, l_search); ++i) {
      //   retset[cur_list_size].id = upper_mem_tags[i];
      //   retset[cur_list_size].distance = dist_scratch[i];
      //   retset[cur_list_size++].flag = true;
      //   visited.insert(upper_mem_tags[i]);
      // }
    } else { 
      compute_and_add_to_retset(&best_medoid, 1);
    }    

    while (!stop_flag) {
      indices.resize(l_search);
      distances.resize(l_search);

      _u64 cur_bw =
          min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
      cur_bw = (cur_bw > 100) ? 100 : cur_bw;

      for (auto &x : distances)
        x = std::numeric_limits<float>::max();
      
      this->page_search_interim(l_search, mem_L, l_search, indices.data(),
                               distances.data(), cur_bw,
                               std::numeric_limits<_u32>::max(),
                               false, page_search_use_ratio, stats, &persist_data);

      for (_u32 i = 0; i < l_search; i++) {
        if (distances[i] > (float) range) {
          res_count = i;
          break;
        } else if (i == l_search - 1)
          res_count = l_search;
      }
      if (res_count < (_u32)(l_search / 2.0))
        stop_flag = true;
      
      retset.resize(2*l_search+1);
      size_t moved_num = kicked.move_to(retset, cur_list_size, l_search);
      cur_list_size += moved_num;

      l_search = l_search * 2;
      if (l_search > max_l_search)
        stop_flag = true;
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }

    this->thread_data.push(persist_data.thread_data);
    this->thread_data.push_notify_all();

    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  char *PQFlashIndex<T>::getHeaderBytes() {
    IOContext & ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *) readReq.buf;
  }
#endif

  // instantiations
  template class PQFlashIndex<_u8>;
  template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;
} // namespace diskann