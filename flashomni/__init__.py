"""
Copyright (c) 2023 by FlashInfer team.

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

from ._build_meta import __version__ as __version__
from .attention import *

# from .attention_processor import _compute_sparse_info_indptr, _compute_sparse_kv_info_indptr 

from .gemm import (
    flashomni_gemm as flashomni_gemm,
    flashomni_gemm_reduction as flashomni_gemm_reduction,
    )

from .quantization import packbits as packbits
from .quantization import segment_packbits as segment_packbits