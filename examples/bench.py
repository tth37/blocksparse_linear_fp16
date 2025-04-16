"""
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 5120)
    (layers): ModuleList(
      (0-47): 48 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
          (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((5120,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((5120,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((5120,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=5120, out_features=152064, bias=False)
)
"""



import torch
import torch.nn as nn
from blocksparse_linear_fp16 import BlockSparseTopKLinear, BlockSparseThresLinear
from tabulate import tabulate

BLOCK_SIZE = (32, 32)
SPARSITY_RATIO = 0.3
BATCH_SIZE = 64

CONFIGS = [
    {
        "title": "q_proj/o_proj: in=5120, out=5120",
        "in_features": 5120,
        "out_features": 5120,
    },
    {
        "title": "k_proj/v_proj: in=5120, out=1024",
        "in_features": 5120,
        "out_features": 1024,
    },
    {
        "title": "gate_proj/up_proj: in=5120, out=13824",
        "in_features": 5120,
        "out_features": 13824,
    },
    {
        "title": "down_proj: in=13824, out=5120",
        "in_features": 13824,
        "out_features": 5120,
    },
]

def generate_dummy_inputs(batch_size, in_features, num_inputs):
    return [
        torch.randn(batch_size, 1, in_features, device="cuda", dtype=torch.float16)
        for _ in range(num_inputs)
    ]

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

def bench_config(config):
    in_features = config["in_features"]
    out_features = config["out_features"]

    blocksparse_topk_linear = BlockSparseTopKLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=BLOCK_SIZE,
        sparsity_ratio=SPARSITY_RATIO,
        device="cuda",
        dtype=torch.float16
    )

    dummy_inputs = generate_dummy_inputs(BATCH_SIZE, in_features, 1)
    blocksparse_topk_linear.run_dummy_inputs(dummy_inputs)

    blocksparse_thres_linear = BlockSparseThresLinear.from_topk_linear(
        blocksparse_topk_linear,
        profile=False
    )

    dense_linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        device="cuda",
        dtype=torch.float16
    )

    sparse_latency = bench(lambda: blocksparse_thres_linear(dummy_inputs[0]))
    dense_latency = bench(lambda: dense_linear(dummy_inputs[0]))

    return sparse_latency, dense_latency



# Benchmark all configurations
results = []
for config in CONFIGS:
    sparse_time, dense_time = bench_config(config)
    results.append({
        "Configuration": config["title"],
        "BlockSparse (ms)": f"{sparse_time:.3f}",
        "Dense (ms)": f"{dense_time:.3f}",
        "Speedup": f"{dense_time/sparse_time:.2f}x"
    })

# Print results table
headers = ["Configuration", "BlockSparse (ms)", "Dense (ms)", "Speedup"]
table = tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid")
print(table)