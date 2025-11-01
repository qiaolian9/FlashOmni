import logging
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from .env import CUTLASS_INCLUDE_DIRS as CUTLASS_INCLUDE_DIRS
from .env import FLASHOMNI_CSRC_DIR as FLASHOMNI_CSRC_DIR
from .env import FLASHOMNI_GEN_SRC_DIR as FLASHOMNI_GEN_SRC_DIR
from .env import FLASHOMNI_INCLUDE_DIR as FLASHOMNI_INCLUDE_DIR
from .env import FLASHOMNI_JIT_DIR as FLASHOMNI_JIT_DIR
from .env import FLASHOMNI_WORKSPACE_DIR as FLASHOMNI_WORKSPACE_DIR

os.makedirs(FLASHOMNI_WORKSPACE_DIR, exist_ok=True)
os.makedirs(FLASHOMNI_CSRC_DIR, exist_ok=True)


class FLASHOMNIJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = FLASHOMNI_WORKSPACE_DIR / "FLASHOMNI_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("FLASHOMNI.jit: " + msg)


logger = FLASHOMNIJITLogger("FLASHOMNI.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FLASHOMNI requires sm75+")


def clear_cache_dir():
    if os.path.exists(FLASHOMNI_JIT_DIR):
        import shutil

        shutil.rmtree(FLASHOMNI_JIT_DIR)


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            suppress(ValueError)


remove_unwanted_pytorch_nvcc_flags()

sm90a_nvcc_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags=None,
    extra_include_paths=None,
):
    verbose = os.environ.get("FLASHOMNI_JIT_VERBOSE", "0") == "1"

    if extra_cflags is None:
        extra_cflags = []
    if extra_cuda_cflags is None:
        extra_cuda_cflags = []

    cflags = ["-O3", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads",
        "4",
        "-use_fast_math",
        "-DFLASHOMNI_ENABLE_F16",
        "-DFLASHOMNI_ENABLE_INT8",
        "-DFLASHOMNI_ENABLE_BF16",
        "-DFLASHOMNI_ENABLE_FP8_E4M3",
        "-DFLASHOMNI_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    logger.info(f"Loading JIT ops: {name}")
    check_cuda_arch()
    build_directory = FLASHOMNI_JIT_DIR / name
    if os.path.exists(build_directory):
        import shutil
        logger.info(f"Removing existing build directory: {build_directory}")
        shutil.rmtree(build_directory)
    os.makedirs(build_directory, exist_ok=True)
    if extra_include_paths is None:
        extra_include_paths = []
    extra_include_paths += [
        FLASHOMNI_INCLUDE_DIR,
        FLASHOMNI_CSRC_DIR,
    ] + CUTLASS_INCLUDE_DIRS
    lock = FileLock(FLASHOMNI_JIT_DIR / f"{name}.lock", thread_local=False)
    with lock:
        torch_cpp_ext.load(
            name,
            list(map(lambda _: str(_), sources)),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True,
            # We switched to torch.library, so will be loaded into torch.ops
            # instead of into a separate module.
            is_python_module=False,
        )
    logger.info(f"Finished loading JIT ops: {name}")
    return getattr(torch.ops, name)
