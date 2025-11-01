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
#include "aot_default_additional_params.h"
#include "pytorch_extension_utils.h"
//========== mySparse for diffusion ==========

at::Tensor BatchSparseFAWithKVPlan(
at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
int64_t head_dim_vo);

void BatchSparseFAWithRaggedKVRun(at::Tensor float_workspace_buffer,
        at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
        at::Tensor q, at::Tensor k, at::Tensor v,
        std::optional<at::Tensor> sparse_info, std::optional<at::Tensor> sparse_kv_info,
        std::optional<at::Tensor> sparse_info_indptr, std::optional<at::Tensor> sparse_kv_info_indptr, 
        int64_t sparse_block_size_for_q, int64_t sparse_block_size_for_kv, bool is_full,
        at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o,
        std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
        int64_t layout BATCH_SPARSEFA_ADDITIONAL_FUNC_PARAMS);

// ========== gemm ==========
void FLASHOMNIGEMM(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  std::optional<at::Tensor> sparse_info, std::optional<at::Tensor> sparse_info_indptr, int64_t num_text_tokens, bool is_full);


void FLASHOMNIGEMMReduction(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens,  bool is_for_cached);


//========== quantization ==========

void packbits(at::Tensor x, const std::string& bitorder, at::Tensor y);

void segment_packbits(at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr,
                      const std::string& bitorder, at::Tensor y);

              
//========== Torch Library ==========
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // sparseFA
  m.def("batch_sparseFA_with_kv_plan", BatchSparseFAWithKVPlan);
  m.def("batch_sparseFA_with_ragged_kv_run", BatchSparseFAWithRaggedKVRun);
  // FLASHOMNIGEMM
  m.def("flashomni_gemm", FLASHOMNIGEMM);
  m.def("flashomni_gemm_reduction", FLASHOMNIGEMMReduction);

  // quantization
  // GPU packbits operator
  m.def("packbits", packbits);
  // GPU segment packbits operator
  m.def("segment_packbits", segment_packbits);
}
