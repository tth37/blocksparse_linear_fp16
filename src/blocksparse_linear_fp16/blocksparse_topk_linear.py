import torch
import torch.nn as nn
import math
from .kernels import avg_magnitude, block_mask_topk, blocksparse_masked_gemm

class BlockSparseTopKLinear(nn.Module):
    thres: float
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int],
        sparsity_ratio: float,
        profile: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert dtype == torch.float16, "Only float16 is supported"
        self.in_features = in_features
        self.out_features = out_features
        self.block_m, self.block_k = block_size
        self.sparsity_ratio = sparsity_ratio
        self.profile = profile
        self.weight = nn.Parameter(torch.randn((in_features, out_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.thres = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        bsz, seq, hidden_size = x.shape
        x = x.view(bsz * seq, hidden_size)
        if seq != 1:
            return (x @ self.weight).view(bsz, seq, self.out_features) + self.bias
        block_activation = avg_magnitude(
            x, self.block_m, self.block_k
        )
        block_mask = block_mask_topk(
            block_activation, self.sparsity_ratio
        )
        if self.profile:
            # check if all blocks are active
            total_blocks = block_mask.numel()
            active_blocks = block_mask.sum().item()
            if active_blocks == total_blocks:
                self.thres = float("inf")
            else:
                # assert False, "FUCK"
                self.thres = torch.max(
                    block_activation[~block_mask]
                ).item()
        x = blocksparse_masked_gemm(
            x, self.weight, block_mask,
            self.block_m, self.block_k
        )
        x = x.view(bsz, seq, self.out_features)
        return x + self.bias
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
