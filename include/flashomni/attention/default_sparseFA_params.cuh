/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHOMNI_SPARSEFA_PARAMS_CUH_
#define FLASHOMNI_SPARSEFA_PARAMS_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
// #include "../page.cuh"
#include "../fastdiv.cuh"

namespace flashomni {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchRaggedSparseFAParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  // sparse_info
  uint8_t* sparse_info;
  uint8_t* sparse_kv_info;
  IdType* sparse_info_indptr;
  IdType* sparse_kv_info_indptr;
  uint32_t sparse_block_size_for_q;
  uint32_t sparse_block_size_for_kv;
  bool is_full;

  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;  // maybe_q_rope_offset is only used for fused-rope attention
  IdType* maybe_k_rope_offset;  // maybe_k_rope_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ BatchRaggedSparseFAParams()
      : q(nullptr),
        k(nullptr),
        v(nullptr),
        sparse_info(nullptr),
        sparse_kv_info(nullptr),
        sparse_info_indptr(nullptr),
        sparse_kv_info_indptr(nullptr),
        sparse_block_size_for_q(128),
        sparse_block_size_for_kv(128),
        is_full(false),
        maybe_custom_mask(nullptr),
        q_indptr(nullptr),
        kv_indptr(nullptr),
        maybe_mask_indptr(nullptr),
        maybe_q_rope_offset(nullptr),
        maybe_k_rope_offset(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        group_size(),
        num_qo_heads(0),
        num_kv_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        k_stride_n(0),
        k_stride_h(0),
        v_stride_n(0),
        v_stride_h(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ BatchRaggedSparseFAParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, 
                                    uint8_t* sparse_info, uint8_t* sparse_kv_info, 
                                    IdType* sparse_info_indptr, IdType* sparse_kv_info_indptr, 
                                    uint32_t sparse_block_size_for_q, uint32_t sparse_block_size_for_kv,
                                    bool is_full, uint8_t* maybe_custom_mask,
                                    IdType* q_indptr, IdType* kv_indptr, IdType* maybe_mask_indptr,
                                    IdType* maybe_q_rope_offset, IdType* maybe_k_rope_offset,
                                    DTypeO* o, float* lse, float* maybe_alibi_slopes,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                    uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                                    uint32_t kv_stride_h, 
                                    float logits_soft_cap, float sm_scale, float rope_scale,
                                    float rope_theta)
      : q(q),
        k(k),
        v(v),
        sparse_info(sparse_info),
        sparse_kv_info(sparse_kv_info),
        sparse_info_indptr(sparse_info_indptr),
        sparse_kv_info_indptr(sparse_kv_info_indptr),
        sparse_block_size_for_q(sparse_block_size_for_q),
        sparse_block_size_for_kv(sparse_block_size_for_kv),
        is_full(is_full),
        maybe_custom_mask(maybe_custom_mask),
        q_indptr(q_indptr),
        kv_indptr(kv_indptr),
        maybe_mask_indptr(maybe_mask_indptr),
        maybe_q_rope_offset(maybe_q_rope_offset),
        maybe_k_rope_offset(maybe_k_rope_offset),
        o(o),
        lse(lse),
        maybe_alibi_slopes(maybe_alibi_slopes),
        group_size(num_qo_heads / num_kv_heads),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        k_stride_n(kv_stride_n),
        k_stride_h(kv_stride_h),
        v_stride_n(kv_stride_n),
        v_stride_h(kv_stride_h),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};


}  // namespace flashomni

#endif  // FLASHOMNI_DECODE_PARAMS_CUH_
