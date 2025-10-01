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
```

To pre-compile essential kernels ahead-of-time (AOT), run the following command:


### Trying it out

Below is a minimal example of using FlashOmni's sparse attention and linear kernels:

```python
import torch
import flashomni

```



