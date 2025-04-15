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

from .kernels import *

class BlockSparseThresLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int],
        thres: float,
        profile: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_m, self.block_k = block_size
        self.thres = thres
        self.profile = profile
        self.weight = nn.Parameter(torch.randn((in_features, out_features), device=device, dtype=dtype))
        self.sparsity_ratio = 0.0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        bsz, hidden_size = x.shape
        block_mask = avg_magnitude_threshold(
            x, self.block_m, self.block_k, self.thres
        )
        if self.profile:
            total_blocks = block_mask.numel()
            active_blocks = block_mask.sum().item()
            self.sparsity_ratio = (total_blocks - active_blocks) / total_blocks
        x = blocksparse_masked_gemm(
            x, self.weight, block_mask,
            self.block_m, self.block_k
        )
        return x
    

class BlockSparseTopkLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int],
        topk_ratio: float,
        profile: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_m, self.block_k = block_size
        self.topk_ratio = topk_ratio
        self.profile = profile
        self.weight = nn.Parameter(torch.randn((in_features, out_features), device=device, dtype=dtype))
        self.thres = 0.0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        bsz, hidden_size = x.shape
        block_activation = avg_magnitude(
            x, self.block_m, self.block_k
        )
        block_mask = block_mask_topk(
            block_activation, self.topk_ratio
        )
        if self.profile:
            self.thres = torch.max(
                block_activation[~block_mask]
            ).item()
        x = blocksparse_masked_gemm(
            x, self.weight, block_mask,
            self.block_m, self.block_k
        )
        return x
# _block_sparse_linear = BlockSparseLinear(4096, 14336*2, (32, 32), 0.8, "cuda", torch.float16)
# block_sparse_linear = torch.compile(_block_sparse_linear)
# _dense_linear = nn.Linear(4096, 14336*2).cuda().half()
# dense_linear = torch.compile(_dense_linear)

# def bench(fn, warmup=20, iter=100):
#     for _ in range(warmup):
#         fn()

#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     start_event.record()
#     for _ in range(iter):
#         fn()
#     end_event.record()
#     torch.cuda.synchronize()

#     elapsed_time = start_event.elapsed_time(end_event)
#     return elapsed_time / iter

# x = torch.randn(32, 4096).cuda().half()

# block_sparse_latency = bench(lambda: block_sparse_linear(x))
# dense_latency = bench(lambda: dense_linear(x))
# _, sparsity_ratio = _block_sparse_linear(x, profile=True)

# print(f"Block Sparse Latency: {block_sparse_latency:.6f} ms")
# print(f"Dense Latency: {dense_latency:.6f} ms")
# print(f"Sparsity Ratio: {sparsity_ratio:.6f}")
