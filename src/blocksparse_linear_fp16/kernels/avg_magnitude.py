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
def avg_magnitude_kernel(
    x_ptr, output_ptr,  # Renamed output_mask_ptr to output_ptr
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    block_m: tl.constexpr, block_k: tl.constexpr,
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
    ).to(tl.float32) # Ensure float32 for calculations

    abs_sum = tl.sum(tl.abs(block_data))

    num_valid_elements = tl.minimum(block_m, M - start_m) * tl.minimum(block_k, N - start_n)
    block_mean = abs_sum / tl.maximum(1.0, num_valid_elements.to(tl.float32))

    output_addr = output_ptr + pid_m * stride_outm + pid_n * stride_outn
    tl.store(output_addr, block_mean)


def avg_magnitude(
    x: torch.Tensor, block_m: int, block_k: int
) -> torch.Tensor:
    M, N = x.shape
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_k - 1) // block_k

    output_avg_mag = torch.empty((num_blocks_m, num_blocks_n), dtype=torch.float32, device=x.device)

    grid = (num_blocks_m, num_blocks_n)

    avg_magnitude_kernel[grid](
        x_ptr=x,
        output_ptr=output_avg_mag,
        M=M, N=N,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_outm=output_avg_mag.stride(0), stride_outn=output_avg_mag.stride(1),
        block_m=block_m,
        block_k=block_k,
    )
    return output_avg_mag


def avg_magnitude_ref(
    x: torch.Tensor, block_m: int, block_k: int
) -> torch.Tensor:
    M, N = x.shape
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_k - 1) // block_k

    output_avg_mag = torch.zeros((num_blocks_m, num_blocks_n), dtype=torch.float32, device=x.device)

    for i in range(num_blocks_m):
        for j in range(num_blocks_n):
            r_start = i * block_m
            r_end = min(r_start + block_m, M)
            c_start = j * block_k
            c_end = min(c_start + block_k, N)

            block = x[r_start:r_end, c_start:c_end]

            if block.numel() > 0:
                block_mean = block.abs().float().mean() # Use .float() for consistent mean calculation
            else:
                block_mean = torch.tensor(0.0, dtype=torch.float32, device=x.device) # Or nan

            output_avg_mag[i, j] = block_mean

    return output_avg_mag

def test_avg_magnitude(num_iterations=200):
    for i in tqdm(range(num_iterations), desc="Testing avg_magnitude"):
        M = random.randint(1, 256)
        N = random.randint(2048, 14336)
        block_m = random.choice([16, 32, 64])
        block_n = random.choice([16, 32, 64])
        x = torch.randn((M, N), device="cuda", dtype=torch.float16)
        block_mask = avg_magnitude(x, block_m, block_n)
        block_mask_ref = avg_magnitude_ref(x, block_m, block_n)
        torch.testing.assert_close(block_mask, block_mask_ref)
        


if __name__ == "__main__":
    test_avg_magnitude()