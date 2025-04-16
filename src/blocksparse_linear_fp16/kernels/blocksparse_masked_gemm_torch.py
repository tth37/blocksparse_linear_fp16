import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def blocksparse_masked_gemm_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    block_mask: torch.Tensor, # Expected shape (num_blocks_m, num_blocks_k), dtype=torch.bool or integer
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_k = (K + block_k - 1) // block_k

    M_pad = num_blocks_m * block_m
    K_pad = num_blocks_k * block_k

    pad_k = K_pad - K
    pad_m = M_pad - M
    
    if pad_m > 0 or pad_k > 0:
        a_padded = F.pad(a, (0, pad_k, 0, pad_m), "constant", 0.0)
    else:
        a_padded = a # No padding needed for 'a'

    block_mask_bool = block_mask.to(device=a.device, dtype=torch.bool)
    
    mask_m_expanded = block_mask_bool.repeat_interleave(block_m, dim=0) 
    full_mask = mask_m_expanded.repeat_interleave(block_k, dim=1) 
    
    masked_a_padded = a_padded * full_mask
    masked_a_for_gemm = masked_a_padded[:M, :] # Shape: (M, K_pad)

    if pad_k > 0:
        b_padded = F.pad(b, (0, 0, 0, pad_k), "constant", 0.0) # Shape: (K_pad, N)
    else:
        b_padded = b # No padding needed for 'b'

    result = torch.matmul(masked_a_for_gemm, b_padded)

    return result