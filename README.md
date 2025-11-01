# FlashOmni

FlashOmni is a unified sparse attention engine, which supports diverse sparse methods and reduces corresponding redundent computation, to accelerate Diffusion Transformers.

This repository is the official implementation of <br>
[**FlashOmni: A Unified Sparse Attention Engine for Diffusion Transformers**](https://arxiv.org/abs/2509.25401) <br>
(Liang Qiao, Yue Dai et al;).

**Code will be released soon!!!**

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Install from Source

Alternatively, build FlashOmni from source:

```bash
git clone git@github.com:qiaolian9/FlashOmni.git --recursive
cd FlashOmni
# To pre-compile essential kernels ahead-of-time (AOT), run the following command:
FLASHOMNI_ENABLE_AOT=1 bear -- pip install -e . -v
```

To pre-compile essential kernels ahead-of-time (AOT), run the following command:


### Trying it out

Below is a minimal example of using FlashOmni's sparse attention and linear kernels (details refers to ./benchmark):

```python
import torch
import flashomni
# sparse info: (sparse_info, sparse_kv_info, sparse_info_indptr, sparse_kv_info_indptr)
# attention
wrapper = flashomni.attention.BatchFlashOmniFAWithRaggedKVWrapper(
        workspace_buffer, kv_layout
    )
wrapper.plan(...)
wrapper.run(q, k, v, sparse_info, sparse_kv_info, sparse_info_indptr, sparse_kv_info_indptr)
# gemm-q
out_flashomni_linear = flashomni.flashomni_gemm(...)
# gemm-o
out_flashomni_gemm_reduction = flashomni.flashomni_gemm_reduction(..., is_for_cache=True)
out_flashomni_gemm_reduction_all = flashomni.flashomni_gemm_reduction(..., is_for_cache=False)
        
        
```

## Citation
A paper describing FlashOmni's techniques is available [on arxiv]([https://dl.acm.org/doi/abs/10.1145/3676641.3716269](https://arxiv.org/abs/2509.25401)). Please cite FlashOmni as:

``` bibtex
@article{qiao2025flashomni,
  title={FlashOmni: A Unified Sparse Attention Engine for Diffusion Transformers},
  author={Qiao, Liang and Dai, Yue and Huang, Yeqi and Kan, Hongyu and Shi, Jun and An, Hong},
  journal={arXiv preprint arXiv:2509.25401},
  year={2025}
}
```



