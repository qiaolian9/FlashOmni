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
#include <cstdint>
#include <flashomni/gemm/w16a16/gemm.cuh>

#include "pytorch_extension_utils.h"

using namespace flashomni;
using namespace flashomni::gemm;

void FLASHOMNIGEMM(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size,
              int64_t num_qo_heads, std::optional<at::Tensor> sparse_info, std::optional<at::Tensor> sparse_info_indptr, int64_t num_text_tokens, bool is_full) {
  unsigned int batch_size = x_ptr.size(0);
  unsigned int M = x_ptr.size(1);
  unsigned int K = x_ptr.size(2);
  unsigned int N = w_ptr.size(0); // weight = (N, K)
  unsigned int head_dim = N / static_cast<uint32_t>(num_qo_heads);

  TORCH_CHECK(x_ptr.size(0) == y_ptr.size(0), "Batch sizes must match");
  TORCH_CHECK(x_ptr.size(1) == y_ptr.size(1), "Token sizes must match");
  TORCH_CHECK(x_ptr.size(2) == w_ptr.size(1) && w_ptr.size(0) == y_ptr.size(2),
              "Result tensor has incorrect shape");
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "only support M % 128 == 0, N % 128 == 0, K % 32 == 0");
  // x_ptr.dtype();
  
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(x_ptr.scalar_type(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    cutlass_t* bias_ptr = bias.has_value()? 
             reinterpret_cast<cutlass_t*>(bias.value().data_ptr())  : nullptr;
    uint8_t* sparse_info_ptr = sparse_info.has_value()? 
             reinterpret_cast<uint8_t*>(sparse_info.value().data_ptr())  : nullptr;
    uint32_t* sparse_info_indptr_ptr = sparse_info_indptr.has_value()? 
             reinterpret_cast<uint32_t*>(sparse_info_indptr.value().data_ptr())  : nullptr;
    auto status = FLASHOMNIGEMMRun<cutlass_t>(
        reinterpret_cast<cutlass_t*>(x_ptr.data_ptr()), reinterpret_cast<cutlass_t*>(w_ptr.data_ptr()), 
        reinterpret_cast<cutlass_t*>(y_ptr.data_ptr()),  bias_ptr,
        M, N, K, sparse_q_size, batch_size, num_qo_heads, head_dim, sparse_info_ptr, 
        sparse_info_indptr_ptr, num_text_tokens, is_full, stream);
    TORCH_CHECK(status == cudaSuccess,
          "Failed to run FLASHOMNIGEMM: ", cudaGetErrorString(status));
    return true;
  });

}
