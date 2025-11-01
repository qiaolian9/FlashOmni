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

import argparse
from itertools import product
from pathlib import Path
from typing import List

from . import (
    generate_aot_default_additional_params_header,
    generate_batch_ragged_sparseFA_inst,
)


def get_instantiation_cu(args: argparse.Namespace) -> List[str]:
    def write_if_different(path: Path, content: str) -> None:
        # print(f"path: {path}")
        if path.exists() and path.read_text() == content:
            return
        print("write")
        path.write_text(content)

    path: Path = args.path
    head_dims: List[int] = args.head_dims
    pos_encoding_modes: List[int] = args.pos_encoding_modes
    use_fp16_qk_reductions: List[int] = args.use_fp16_qk_reductions
    mask_modes: List[int] = args.mask_modes
    enable_f16: bool = args.enable_f16
    enable_bf16: bool = args.enable_bf16
    enable_fp8_e4m3: bool = args.enable_fp8_e4m3
    enable_fp8_e5m2: bool = args.enable_fp8_e5m2

    path.mkdir(parents=True, exist_ok=True)

    write_if_different(
        path / "aot_default_additional_params.h",
        generate_aot_default_additional_params_header.get_aot_default_additional_params_header_str(),
    )

    idtypes = ["i32"]
    prefill_dtypes = []
    decode_dtypes = []
    fp16_dtypes = []
    fp8_dtypes = []
    if enable_f16:
        prefill_dtypes.append("f16")
        decode_dtypes.append("f16")
        fp16_dtypes.append("f16")
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")
    if enable_fp8_e4m3:
        fp8_dtypes.extend(["e4m3"])
        decode_dtypes.extend(["e4m3"])
    if enable_fp8_e5m2:
        fp8_dtypes.extend(["e5m2"])
        decode_dtypes.extend(["e5m2"])


    # batch sparseFA files
    batch_sparseFA_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        use_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in product(
        head_dims,
        pos_encoding_modes,
        use_fp16_qk_reductions,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            product(prefill_dtypes, fp8_dtypes)
        ):

            fname = f"batch_ragged_sparseFA_head_qk_{head_dim}_head_vo_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{use_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            content = generate_batch_ragged_sparseFA_inst.get_cu_file_str(
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            for logits_soft_cap in [True, False]:
                if (
                    mask_mode == 0
                ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                    batch_sparseFA_uris.append(
                        f"batch_sparseFA_with_kv_dtype_q_{dtype_q}_"
                        f"dtype_kv_{dtype_kv}_"
                        f"dtype_o_{dtype_q}_"
                        f"dtype_idx_{idtype}_"
                        f"head_dim_qk_{head_dim}_"
                        f"head_dim_vo_{head_dim}_"
                        f"posenc_{pos_encoding_mode}_"
                        f"use_logits_cap_{logits_soft_cap}_"
                        f"f16qk_{bool(use_fp16_qk_reduction)}"
                    )

    return (
        batch_sparseFA_uris
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate cuda files")
    parser.add_argument(
        "--path", type=Path, required=True, help="Path to the dispatch inc file"
    )
    parser.add_argument(
        "--head_dims", type=int, required=True, nargs="+", help="Head dimensions"
    )
    parser.add_argument(
        "--pos_encoding_modes",
        type=int,
        required=True,
        nargs="+",
        help="Position encoding modes",
    )
    parser.add_argument(
        "--use_fp16_qk_reductions",
        type=lambda x: (
            x if isinstance(x, int) else int(x.lower() == "true" or x.lower() == "on")
        ),
        required=True,
        nargs="+",
        help="Allow fp16 qk reductions",
    )
    parser.add_argument(
        "--mask_modes",
        type=int,
        required=True,
        nargs="+",
        help="Mask modes",
    )
    parser.add_argument(
        "--enable_f16",
        type=lambda x: (
            x if isinstance(x, int) else (x.lower() == "true" or x.lower() == "on")
        ),
        required=True,
        nargs="?",
        help="Enable fp16",
    )
    parser.add_argument(
        "--enable_bf16",
        type=lambda x: (
            x if isinstance(x, int) else (x.lower() == "true" or x.lower() == "on")
        ),
        required=True,
        nargs="?",
        help="Enable bf16",
    )
    parser.add_argument(
        "--enable_fp8_e4m3",
        type=lambda x: (
            x if isinstance(x, int) else (x.lower() == "true" or x.lower() == "on")
        ),
        default=True,
        nargs="?",
        help="Enable fp8_e4m3",
    )
    parser.add_argument(
        "--enable_fp8_e5m2",
        type=lambda x: (
            x if isinstance(x, int) else (x.lower() == "true" or x.lower() == "on")
        ),
        default=True,
        nargs="?",
        help="Enable fp8_e5m2",
    )
    args = parser.parse_args()
    get_instantiation_cu(args)
