import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
import torch.nn.functional as F
import functools
import itertools
import random
from tqdm import tqdm
import os
import json

def get_gpu_device_name():
    try:
        device_id = torch.cuda.current_device()
        return torch.cuda.get_device_name(device_id).replace(" ", "_")
    except Exception as e:
        print(f"Warning: Could not get GPU device name: {e}. Using 'Unknown_GPU'.")
        return "Unknown_GPU"
    
CACHE_DIR = os.path.expanduser("~/.cache/blocksparse_linear_fp16/")

def save_best_config(M, N, K, block_M, block_K, best_config):
    gpu_name = get_gpu_device_name()
    config_key = f"M{M}_N{N}_K{K}_blockM{block_M}_blockK{block_K}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    config_file_path = os.path.join(CACHE_DIR, f"{gpu_name}.json")

    configs = {}
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                configs = json.load(f)
        except Exception as e:
            configs = {}

    configs[config_key] = list(best_config) 

    try:
        with open(config_file_path, 'w') as f:
            json.dump(configs, f, indent=4)
    except IOError as e:
        print(f"Warning: Could not write cache file {config_file_path}. Error: {e}")

def load_best_config(M, N, K, block_M, block_K):
    gpu_name = get_gpu_device_name()
    config_key = f"M{M}_N{N}_K{K}_blockM{block_M}_blockK{block_K}"
    
    config_file_path = os.path.join(CACHE_DIR, f"{gpu_name}.json")
    
    if not os.path.exists(config_file_path):
        return None
        
    try:
        with open(config_file_path, 'r') as f:
            configs = json.load(f)
            config_list = configs.get(config_key)
            if config_list:
                return tuple(config_list) 
            else:
                return None
    except Exception as e:
        return None

def blocksparse_masked_gemm_kernel(
    M, N, K, block_M, block_N, block_K,
    num_stages, thread_num, enable_rasteration,
    dtype="float16", accum_dtype="float"
):
    block_mask_shape = (T.ceildiv(M, block_M), T.ceildiv(K, block_K))

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if BlockMask[by, k]:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


@functools.cache
def get_best_config(M, N, K, block_M, block_K):
    def get_configs():
        block_N = [64, 128, 256]
        num_stages = [1, 2, 3]
        thread_num = [128, 256]
        enable_rasteration = [True, False]

        _configs = list(
            itertools.product(block_N, num_stages, thread_num, enable_rasteration)
        )

        return [{
            "block_N": c[0],
            "num_stages": c[1],
            "thread_num": c[2],
            "enable_rasteration": c[3]
        } for c in _configs]
    
    def kernel(
        block_N=None, num_stages=None, thread_num=None, enable_rasteration=None
    ):
        return blocksparse_masked_gemm_kernel(
            M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration
        )
    
    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(),
    ).set_compile_args(
        out_idx=[-1],
        ref_prog=lambda A, B, BlockMask: A @ B,
        skip_check=True,
        cache_input_tensors=False,
        target="auto"
    )

    result = autotuner.run()
    print("best config:", result.config)
    print("best latency:", result.latency)
    print("ref latency:", result.ref_latency)
    return result.config

@functools.cache
def get_tuned_kernel(M, N, K, block_M, block_K):
    best_config = load_best_config(M, N, K, block_M, block_K)
    if best_config is None:
        best_config = get_best_config(M, N, K, block_M, block_K)
        save_best_config(M, N, K, block_M, block_K, best_config)
    func = blocksparse_masked_gemm_kernel(
        M, N, K, block_M=block_M, block_N=best_config[0], block_K=block_K,
        num_stages=best_config[1], thread_num=best_config[2],
        enable_rasteration=best_config[3]
    )
    kernel = tilelang.compile(func, out_idx=-1)
    return kernel

def blocksparse_masked_gemm_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    return get_tuned_kernel(M, N, K, block_m, block_k)(
        a, b, block_mask,
    )