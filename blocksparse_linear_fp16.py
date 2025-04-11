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


@triton.jit
def avg_magnitude_threshold_kernel(
    x_ptr, output_mask_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    block_m: tl.constexpr, block_k: tl.constexpr,
    thres: float,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_m = pid_m * block_m
    start_n = pid_n * block_k

    offs_m = start_m + tl.arange(0, block_m) # Shape (block_m,)
    offs_n = start_n + tl.arange(0, block_k) # Shape (block_k,)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn

    mask_m = offs_m < M # Shape (block_m,)
    mask_n = offs_n < N # Shape (block_k,)
    block_load_mask = mask_m[:, None] & mask_n[None, :] # Shape (block_m, block_k)

    block_data = tl.load(
        x_ptrs,
        mask=block_load_mask,
        other=0.0,
    ).to(tl.float32)

    abs_sum = tl.sum(tl.abs(block_data))
    num_valid_elements = tl.minimum(block_m, M - start_m) * tl.minimum(block_k, N - start_n)

    block_mean = abs_sum / tl.maximum(1.0, num_valid_elements.to(tl.float32))
    result = block_mean > thres

    output_ptr = output_mask_ptr + pid_m * stride_outm + pid_n * stride_outn
    tl.store(output_ptr, result)

def avg_magnitude_threshold(
    x: torch.Tensor, block_m: int, block_k: int, thres: float
) -> torch.Tensor:
    M, N = x.shape

    # num_blocks_m = T.ceildiv(M, block_m)
    # num_blocks_n = T.ceildiv(N, block_k)
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_k - 1) // block_k

    output_mask = torch.empty((num_blocks_m, num_blocks_n), dtype=torch.bool, device=x.device)

    grid = (num_blocks_m, num_blocks_n)

    avg_magnitude_threshold_kernel[grid](
        x_ptr=x,
        output_mask_ptr=output_mask,
        M=M, N=N,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_outm=output_mask.stride(0), stride_outn=output_mask.stride(1),
        block_m=block_m,
        block_k=block_k,
        thres=thres,
    )

    return output_mask

def avg_magnitude_threshold_ref(
    x: torch.Tensor, block_m: int, block_k: int, thres: float
) -> torch.Tensor:
    M, N = x.shape
    # num_blocks_m = T.ceildiv(M, block_m)
    # num_blocks_n = T.ceildiv(N, block_k)
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_k - 1) // block_k
    output_mask = torch.zeros((num_blocks_m, num_blocks_n), dtype=torch.bool, device=x.device)

    for i in range(num_blocks_m):
        for j in range(num_blocks_n):
            r_start = i * block_m
            r_end = min(r_start + block_m, M)
            c_start = j * block_k
            c_end = min(c_start + block_k, N)

            block = x[r_start:r_end, c_start:c_end].to(torch.float32)
            block_mean = block.abs().mean()
            output_mask[i, j] = block_mean > thres

    return output_mask

def test_avg_magnitude_threshold(num_iterations=200):
    for i in tqdm(range(num_iterations), desc="Testing avg_magnitude_threshold"):
        M = random.randint(1, 256)
        N = random.randint(2048, 14336)
        block_m = random.choice([16, 32, 64])
        block_n = random.choice([16, 32, 64])
        thres = random.uniform(-0.5, 0.5)
        x = torch.randn((M, N), device="cuda", dtype=torch.float16)
        block_mask = avg_magnitude_threshold(x, block_m, block_n, thres)
        block_mask_ref = avg_magnitude_threshold_ref(x, block_m, block_n, thres)
        torch.testing.assert_close(block_mask, block_mask_ref)

    print("avg_magnitude_threshold test passed!")

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
    # num_blocks_m = T.ceildiv(M, block_m)
    # num_blocks_k = T.ceildiv(K, block_k)
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
#         M = 35
#         K = 4096
#         N = 14336
#         block_m = random.choice([32])
#         block_k = random.choice([32])
#         num_blocks_m = (M + block_m - 1) // block_m
#         num_blocks_k = (K + block_k - 1) // block_k

#         a = torch.randn((M, K), device="cuda", dtype=torch.float16)
#         b = torch.randn((K, N), device="cuda", dtype=torch.float16)
#         block_mask = torch.randint(0, 2, (num_blocks_m, num_blocks_k), device="cuda", dtype=torch.bool)

#         result = blocksparse_masked_gemm(a, b, block_mask, block_m, block_k)
#         result_ref = blocksparse_masked_gemm_ref(a, b, block_mask, block_m, block_k)

#         torch.testing.assert_close(result, result_ref)

#     print("blocksparse_masked_gemm test passed!")

# test_avg_magnitude_threshold()
# test_blocksparse_masked_gemm()

class BlockSparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int],
        thres: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_m, self.block_k = block_size
        self.thres = thres
        self.weight = nn.Parameter(torch.randn((in_features, out_features), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, profile=False):
        bsz, hidden_size = x.shape
        block_mask = avg_magnitude_threshold(
            x, self.block_m, self.block_k, self.thres
        )
        if profile:
            total_blocks = block_mask.numel()
            active_blocks = block_mask.sum().item()
            sparsity_ratio = (total_blocks - active_blocks) / total_blocks
        x = blocksparse_masked_gemm(
            x, self.weight, block_mask,
            self.block_m, self.block_k
        )
        if profile:
            return x, sparsity_ratio
        return x
    
_block_sparse_linear = BlockSparseLinear(4096, 14336*2, (32, 32), 0.8, "cuda", torch.float16)
block_sparse_linear = torch.compile(_block_sparse_linear)
_dense_linear = nn.Linear(4096, 14336*2).cuda().half()
dense_linear = torch.compile(_dense_linear)

def bench(fn, warmup=20, iter=100):
    for _ in range(warmup):
        fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iter):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    return elapsed_time / iter

x = torch.randn(32, 4096).cuda().half()

block_sparse_latency = bench(lambda: block_sparse_linear(x))
dense_latency = bench(lambda: dense_linear(x))
_, sparsity_ratio = _block_sparse_linear(x, profile=True)

print(f"Block Sparse Latency: {block_sparse_latency:.6f} ms")
print(f"Dense Latency: {dense_latency:.6f} ms")
print(f"Sparsity Ratio: {sparsity_ratio:.6f}")
