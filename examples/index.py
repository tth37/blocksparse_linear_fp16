from blocksparse_linear_fp16 import BlockSparseThresLinear, BlockSparseTopKLinear
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer

blocksparse_topk_linear = BlockSparseTopKLinear(
    4096, 14336, (32, 32), 0.5, device="cuda", dtype=torch.float16
)

x = torch.randn(32, 4096).cuda().half()

blocksparse_topk_linear(x)

# print(blocksparse_topk_linear.thres)
# print("thres", blocksparse_topk_linear.thres)

# blocksparse_thres_linear = BlockSparseThresLinear(
#     4096, 14336, (32, 32), blocksparse_topk_linear.thres, profile=True, device="cuda", dtype=torch.float16
# )
# blocksparse_thres_linear.weight = blocksparse_topk_linear.weight

# x = torch.randn(19, 4096).cuda().half()

# blocksparse_thres_linear(x)

# print("sparsity_ratio", blocksparse_thres_linear.sparsity_ratio)

blocksparse_thres_linear = BlockSparseThresLinear(
    4096, 14336, (32, 32), 0.796, profile=False, device="cuda", dtype=torch.float16
)

dense_linear = nn.Linear(
    4096, 14336, device="cuda", dtype=torch.float16
)

def benchmark_latency(
    fn: callable,
    num_warmup: int = 20,
    num_iterations: int = 200,
):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    return elapsed_time / num_iterations
    

sparse_latency = benchmark_latency(lambda: blocksparse_thres_linear(x))
print("Sparse latency:", sparse_latency)
dense_latency = benchmark_latency(lambda: dense_linear(x))
print("Dense latency:", dense_latency)