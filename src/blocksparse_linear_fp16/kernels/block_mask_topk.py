import torch
import math


def block_mask_topk(
    block_avg_magnitude: torch.Tensor, sparsity_ratio: float
) -> torch.Tensor:
    topk_ratio = 1. - sparsity_ratio
    device = block_avg_magnitude.device
    num_elements = block_avg_magnitude.numel()
    k = min(max(0, math.ceil(num_elements * topk_ratio)), num_elements)

    if k == 0:
        return torch.zeros_like(block_avg_magnitude, dtype=torch.bool, device=device)
    if k == num_elements:
        return torch.ones_like(block_avg_magnitude, dtype=torch.bool, device=device)
    
    flat_tensor = block_avg_magnitude.flatten()
    _, topk_indices = torch.topk(
        flat_tensor, k, largest=True
    )
    block_mask = torch.zeros_like(flat_tensor, dtype=torch.bool, device=device)
    block_mask[topk_indices] = True

    block_mask = block_mask.view_as(block_avg_magnitude)
    return block_mask
