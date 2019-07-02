import platform
import subprocess

import renom


def get_os_version():
    return platform.platform()


def get_python_version():
    return platform.python_version()


def get_gpu_info():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"]).decode("utf-8")
    except Exception as e:
        return "nvidia-smi command is not find."


def get_cuda_version():
    try:
        return subprocess.check_output(["nvcc", "-V"]).decode("utf-8")
    except Exception as e:
        return "nvcc command is not find."


def get_renom_version():
    return renom.__version__
