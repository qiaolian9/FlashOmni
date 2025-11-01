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
#include "pytorch_extension_utils.h"

void FLASHOMNIGEMM(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  std::optional<at::Tensor> sparse_info, std::optional<at::Tensor> sparse_info_indptr, int64_t num_text_tokens, bool is_full);

void FLASHOMNIGEMMReduction(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens, bool is_for_cached);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("flashomni_gemm", FLASHOMNIGEMM);
  m.def("flashomni_gemm_reduction", FLASHOMNIGEMMReduction);
}
