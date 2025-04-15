import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
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
def get_tuned_kernel(M, N, K, block_M, block_K):
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
    best_config = result.config
    save_best_config(M, N, K, block_M, block_K, best_config)    
    return result.kernel

@functools.cache
def get_cached_kernel(M, N, K, block_M, block_K):
    best_config = load_best_config(M, N, K, block_M, block_K)
    if best_config is None:
        return get_tuned_kernel(M, N, K, block_M, block_K)
    func = blocksparse_masked_gemm_kernel(
        M, N, K, block_M=block_M, block_N=best_config[0], block_K=block_K,
        num_stages=best_config[1], thread_num=best_config[2],
        enable_rasteration=best_config[3]
    )
    kernel = tilelang.compile(func, out_idx=-1)
    return kernel

def blocksparse_masked_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    return get_cached_kernel(M, N, K, block_m, block_k)(
        a, b, block_mask
    )

def blocksparse_masked_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_k = (K + block_k - 1) // block_k
    masked_a = a.clone()

    for i in range(num_blocks_m):
        for j in range(num_blocks_k):
            r_start = i * block_m
            r_end = min(r_start + block_m, M)
            c_start = j * block_k
            c_end = min(c_start + block_k, K)

            if not block_mask[i, j]:
                masked_a[r_start:r_end, c_start:c_end] = 0.0
    
    return masked_a @ b

def test_blocksparse_masked_gemm(num_iterations=200):
    for i in tqdm(range(num_iterations), desc="Testing blocksparse_masked_gemm"):
        M = 35
        K = 4096
        N = 14336
        block_m = random.choice([32])
        block_k = random.choice([32])
        num_blocks_m = (M + block_m - 1) // block_m
        num_blocks_k = (K + block_k - 1) // block_k

        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        block_mask = torch.randint(0, 2, (num_blocks_m, num_blocks_k), device="cuda", dtype=torch.bool)

        result = blocksparse_masked_gemm(a, b, block_mask, block_m, block_k)
        result_ref = blocksparse_masked_gemm_ref(a, b, block_mask, block_m, block_k)

        torch.testing.assert_close(result, result_ref)

    print("blocksparse_masked_gemm test passed!")

# test_blocksparse_masked_gemm(1)