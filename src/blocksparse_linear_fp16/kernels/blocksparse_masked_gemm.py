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
from typing import Union, Literal
from .blocksparse_masked_gemm_tilelang import blocksparse_masked_gemm_tilelang
from .blocksparse_masked_gemm_torch import blocksparse_masked_gemm_torch


def blocksparse_masked_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor,
    block_m: int,
    block_k: int,
    backend: Union[Literal["auto"], Literal["torch"], Literal["tilelang"]] = "auto",
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    if backend == "auto":
        if (block_m < 16 or block_k < 16):
            backend = "torch"
        else:
            backend = "tilelang"

    if backend == "torch":
        return blocksparse_masked_gemm_torch(
            a, b, block_mask, block_m, block_k
        )

    return blocksparse_masked_gemm_tilelang(
        a, b, block_mask, block_m, block_k
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

# def test_blocksparse_masked_gemm(num_iterations=200):
#     for i in tqdm(range(num_iterations), desc="Testing blocksparse_masked_gemm"):
#         M = 358
#         K = 4096
#         N = 14336
#         block_m = random.choice([32])
#         block_k = random.choice([32])
#         num_blocks_m = (M + block_m - 1) // block_m
#         num_blocks_k = (K + block_k - 1) // block_k

#         a = torch.randn((M, K), device="cuda", dtype=torch.float16)
#         b = torch.randn((K, N), device="cuda", dtype=torch.float16)
#         block_mask = torch.randint(0, 2, (num_blocks_m, num_blocks_k), device="cuda", dtype=torch.bool)

#         result = blocksparse_masked_gemm_torch(a, b, block_mask, block_m, block_k)
#         result_ref = blocksparse_masked_gemm_ref(a, b, block_mask, block_m, block_k)

#         torch.testing.assert_close(result, result_ref)

#     print("blocksparse_masked_gemm test passed!")

# test_blocksparse_masked_gemm(20)