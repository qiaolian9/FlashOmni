from .core import *
from .env import*
import ctypes
import os


cuda_lib_path = os.environ.get(
    "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
)
if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
    ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)



try:
    from .. import flashomni_kernels  # noqa: F401
    from .aot_config import prebuilt_ops_uri as prebuilt_ops_uri

    has_prebuilt_ops = True
except ImportError as e:
    if "undefined symbol" in str(e):
        raise ImportError("Loading prebuilt ops failed.") from e

    from .core import logger

    logger.info("Prebuilt kernels not found, using JIT backend")
    prebuilt_ops_uri = {}
    has_prebuilt_ops = False