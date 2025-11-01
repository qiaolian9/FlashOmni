"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .jit import (
    FLASHOMNI_CSRC_DIR,
    has_prebuilt_ops,
    load_cuda_ops
)

from .utils import (
    _get_cache_buf,
    register_custom_op,
    register_fake_op,
)

_gemm_module = None


def get_gemm_module():
    global _gemm_module
    if _gemm_module is None:
        if has_prebuilt_ops:
            _kernels = torch.ops.flashomni_kernels

            module = _kernels
        else:
            print(FLASHOMNI_CSRC_DIR)
            module = load_cuda_ops(
                "gemm",
                [
                    FLASHOMNI_CSRC_DIR / "gemm.cu",
                    FLASHOMNI_CSRC_DIR / "gemm_reduction.cu",
                    FLASHOMNI_CSRC_DIR / "flashomni_gemm_ops.cu",
                ],
                extra_ldflags=["-lcublas", "-lcublasLt"],
            )

        # ====== FP16 & BF16 ======
        # torch library for flashomni_gemm
        @register_custom_op(
            "flashomni::flashomni_gemm", mutates_args=("bias")
        )
        def flashomni_gemm(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            sparse_q_size: int = 128,
            is_full: bool = False,
        ) -> None:
            # print(is_full)
            module.flashomni_gemm.default(
                A,
                B,
                out,
                bias,
                sparse_q_size,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                num_text_tokens,
                is_full,
            )

        @register_fake_op("flashomni::flashomni_gemm")
        def _fake_flashomni_gemm(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            sparse_q_size: int = 128,
            is_full: bool = False,
        ) -> None:
            pass

        # torch library for flashomni_gemm
        @register_custom_op(
            "flashomni::flashomni_gemm_reduction", mutates_args=("bias")
        )
        def flashomni_gemm_reduction(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            is_for_cache: bool = False,
            sparse_q_size: int = 128,
        ) -> None:
            module.flashomni_gemm_reduction.default(
                A,
                B,
                out,
                bias,
                sparse_q_size,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                num_text_tokens,
                is_for_cache
            )

        @register_fake_op("flashomni::flashomni_gemm_reduction")
        def _fake_flashomni_gemm_reduction(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            is_for_cache: bool = False,
            sparse_q_size: int = 128,
        ) -> None:
            pass

        # Register the module
        _gemm_module = SimpleNamespace(
            flashomni_gemm = flashomni_gemm,
            flashomni_gemm_reduction = flashomni_gemm_reduction,
        )

    return _gemm_module


# ====== FP16 & BF16 ======
def flashomni_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    num_qo_heads: int,
    sparse_info: torch.Tensor = None,
    sparse_info_indptr: torch.Tensor = None,
    num_text_tokens: int = 512,
    bias: torch.Tensor = None,
    out: torch.Tensor = None,
    sparse_q_size: int = 128,
    is_full: bool = False,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[0]),
            device=A.device,
            dtype=A.dtype,
        )
    get_gemm_module().flashomni_gemm(
        A,
        B,
        out,
        num_qo_heads,
        sparse_info,
        sparse_info_indptr,
        num_text_tokens,
        bias,
        sparse_q_size=sparse_q_size,
        is_full=is_full,
    )
    return out

def flashomni_gemm_reduction(
    A: torch.Tensor,
    B: torch.Tensor,
    num_qo_heads: int,
    sparse_info: torch.Tensor,
    sparse_info_indptr: torch.Tensor,
    num_text_tokens: int,
    bias: torch.Tensor = None,
    is_for_cache: bool = False,
    sparse_q_size: int = 128,
) -> torch.Tensor:
    out = torch.empty(
        (A.shape[0], A.shape[1], B.shape[0]),
        device=A.device,
        dtype=A.dtype,
    )
    get_gemm_module().flashomni_gemm_reduction(
        A,
        B,
        out,
        num_qo_heads,
        sparse_info,
        sparse_info_indptr,
        num_text_tokens,
        bias=bias,
        is_for_cache=is_for_cache,
        sparse_q_size=sparse_q_size,
    )
    return out
