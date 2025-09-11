# FlashOmni

FlashOmni is an unified sparse attention engine, which supports diverse sparse methods and reduces corresponding redundent computation, to accelerate Diffusion Transformers.

This repository is the official implementation of <br>
[**FlashOmni: An Unified Sparse Attention Engine to Accelerate Diffusion Transformers**]() <br>
(Liang Qiao, Yue Dai et al;).


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


## Citation
If you find FlashOmni helpful in your project or research, please consider citing our [paper]():Please cite FlashOmni as:

``` bibtex
@inproceedings{qiao2025flashomni,
  title={FlashOmni: An Unified Sparse Attention Engine to Accelerate Diffusion Transformers},
  author={Qiao, Liang and Yue, Dai and others},
}
```


