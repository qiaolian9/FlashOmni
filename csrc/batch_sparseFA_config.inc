/*
 * Copyright (c) 2025 by FlashInfer team.
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
// NOTE(Zihao): this is the include file for AOT mode
#pragma once
#include <flashomni/attention/default_sparseFA_params.cuh>
#include <flashomni/attention/variants.cuh>
#include <flashomni/layout.cuh>
#include <flashomni/math.cuh>
// #include <flashomni/page.cuh>
#include <flashomni/utils.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using IdType = int32_t;

#define ADDITIONAL_FUNC_PARAMS BATCH_SPARSEFA_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER BATCH_SPARSEFA_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,    \
                         POS_ENCODING_MODE, USE_LOGITS_SOFT_CAP,              \
                         USE_FP16_QK_REDUCTION, AttentionVariant, RaggedSparseFAParams, ...) \
  {                                                                                               \
    DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {                                                \
      return DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(                                                 \
          q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] {                                   \
            using DTypeO = DTypeQ;                                                                \
            using RaggedSparseFAParams = BatchRaggedSparseFAParams<DTypeQ, DTypeKV, DTypeO, IdType>;       \
            constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                            \
            constexpr bool USE_FP16_QK_REDUCTION = false;                                         \
            constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;                      \
            return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {                              \
              [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                                            \
              return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {              \
                using AttentionVariant =                                                          \
                    DefaultAttention</*use_custom_mask=*/use_custom_mask,                         \
                                      USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;             \
                __VA_ARGS__();                                                                    \
                return true;                                                                      \
              });                                                                                 \
            });                                                                                   \
          });                                                                                     \
    });                                                                                           \
  }
