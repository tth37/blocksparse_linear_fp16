import torch
import torch.nn as nn
import triton
import triton.language as tl
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
import math
import functools
import itertools
import random
from tqdm import tqdm


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

    # return autotuner.run().kernel
    result = autotuner.run()
    print("best config:", result.config)
    return result.kernel

def blocksparse_masked_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    return get_tuned_kernel(M, N, K, block_m, block_k)(
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