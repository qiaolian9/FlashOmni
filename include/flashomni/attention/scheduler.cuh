/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHOMNI_ATTENTION_SCHEDULER_CUH_
#define FLASHOMNI_ATTENTION_SCHEDULER_CUH_

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

#include "../allocator.h"
#include "../exception.h"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "heap.h"

namespace flashomni {
inline auto PrefillBinarySearchKVChunkSize(const bool enable_cuda_graph,
                                           const uint32_t max_batch_size_if_split,
                                           const std::vector<int64_t>& packed_qo_len_arr,
                                           const std::vector<int64_t>& kv_len_arr,
                                           const uint32_t qo_chunk_size,
                                           const uint32_t min_kv_chunk_size = 1) {
  const int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }

  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_len;
  constexpr int64_t min_kv_len = 1;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    int64_t new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                        ceil_div(std::max(kv_len_arr[i], min_kv_len), mid);
    }
    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return std::make_tuple(enable_cuda_graph || low < max_kv_len, low);
}


template <typename IdType>
inline auto PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h,
                                   uint32_t total_num_rows, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, uint32_t max_batch_size_if_split,
                                   bool enable_cuda_graph) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;

  // step 1: determine packed_qo_len_arr and verify qo_indptr contents.
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    if (packed_qo_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHOMNI_ERROR(err_msg.str());
    }
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    if (kv_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_indptr[" << i + 1 << "]" << kv_indptr_h[i + 1] << " - kv_indptr[" << i << "]"
              << kv_indptr_h[i] << " should be non-negative";
      FLASHOMNI_ERROR(err_msg.str());
    }
  }

  // step 2: determine cta_tile_q, kv_chunk_size and total_num_tiles_q
  const uint32_t min_kv_chunk_size = std::max((128 / page_size), 1U);
  uint32_t cta_tile_q;
  uint32_t total_num_tiles_q;
  if (enable_cuda_graph) {
    // When CUDA graphs are enabled, the lengths of sequences determined by
    // qo_indptr_h can vary. We assume that the dummy data based on which
    // the CUDA graph is created fixes the maximum number of tokens.
    const uint64_t max_seq_len = total_num_rows - batch_size + 1;
    uint64_t max_qo_len = uint64_t(max_seq_len) * gqa_group_size;
    cta_tile_q = FA2DetermineCtaTileQ(max_qo_len, head_dim);

    // Find an upper bound for the number of tiles, derived from the total
    // number of rows and the batch size.  The sum of qo lengths rounded
    // up to cta_tile_q will not exceed this number derived from the total
    // number of rows.
    total_num_tiles_q = ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch_size - 1;
  } else {
    int64_t sum_packed_qo_len = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      sum_packed_qo_len += packed_qo_len_arr[i];
    }
    const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);

    total_num_tiles_q = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      total_num_tiles_q += ceil_div(packed_qo_len_arr[i], cta_tile_q);
    }
  }

  auto [split_kv, kv_chunk_size] =
      PrefillBinarySearchKVChunkSize(enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr,
                                     kv_len_arr, cta_tile_q, min_kv_chunk_size);

  // step 3: split qo_indptr and kv_indptr
  uint32_t new_batch_size = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    const int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    const int64_t kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    const int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q);
    const int64_t num_tiles_kv = ceil_div(kv_len, kv_chunk_size);

    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
        new_batch_size += 1;
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  const size_t padded_batch_size =
      enable_cuda_graph ? std::max(max_batch_size_if_split, total_num_tiles_q) : new_batch_size;
  FLASHOMNI_CHECK(new_batch_size <= padded_batch_size,
                   "new batch size should not exceed padded batch size");

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return std::make_tuple(split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size,
                         std::move(request_indices), std::move(qo_tile_indices),
                         std::move(kv_tile_indices), std::move(merge_indptr), std::move(o_indptr));
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t total_num_rows_offset;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        total_num_rows_offset(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            total_num_rows_offset,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 15) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 15, but got " << vec.size();
      FLASHOMNI_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    total_num_rows_offset = vec[2];
    cta_tile_q = vec[3];
    request_indices_offset = vec[4];
    qo_tile_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    merge_indptr_offset = vec[7];
    o_indptr_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    v_offset = vec[10];
    s_offset = vec[11];
    block_valid_mask_offset = vec[12];
    enable_cuda_graph = vec[13];
    split_kv = vec[14];
  }
};

template <typename IdType>
inline cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                               void* int_buffer, void* page_locked_int_buffer,
                               size_t int_workspace_size_in_bytes, PrefillPlanInfo& plan_info,
                               IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t total_num_rows,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t head_dim_qk, uint32_t head_dim_vo, uint32_t page_size,
                               bool enable_cuda_graph, uint32_t sizeof_dtype_o,
                               cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHOMNI_ERROR(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHOMNI_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHOMNI_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size, request_indices_vec,
        qo_tile_indices_vec, kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                             num_kv_heads, head_dim_vo, page_size, max_batch_size_if_split,
                             enable_cuda_graph);

  plan_info.cta_tile_q = cta_tile_q;
  plan_info.total_num_rows = total_num_rows;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.split_kv = split_kv;

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_request_indices");
  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_qo_tile_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * (batch_size + 1),
                                                                 16, "batch_prefill_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");

  if (plan_info.enable_cuda_graph) {
    plan_info.total_num_rows_offset =
        int_allocator.aligned_alloc_offset(sizeof(uint32_t), 16, "batch_prefill_total_num_rows");
    uint32_t* total_num_rows_h =
        GetPtrFromBaseOffset<uint32_t>(page_locked_int_buffer, plan_info.total_num_rows_offset);
    *total_num_rows_h = qo_indptr_h[batch_size];
  }

  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size;
  // std::cout << "kv_chunk_size: " << kv_chunk_size << "\nsplit\t" << split_kv << "\n padded_batchs\t" << padded_batch_size << std::endl;
  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * sizeof(float), 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float), 16, "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");

    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHOMNI_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  return cudaSuccess;
}

inline float cost_function(int qo_len, int kv_len) { return 2 * float(qo_len) + kv_len; }

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec, int size_after_flatten) {
  std::vector<T> result;
  result.reserve(size_after_flatten);
  for (const auto& inner_vec : vec) {
    result.insert(result.end(), inner_vec.begin(), inner_vec.end());
  }
  return result;
}

}  // namespace flashomni
#endif  // FLASHOMNI_ATTENTION_SCHEDULER_CUH_
