from typing import Union, Literal
import torch.nn as nn
from ..blocksparse_topk_linear import BlockSparseTopKLinear
from ..blocksparse_thres_linear import BlockSparseThresLinear

def _replace_linear_with_blocksparse_topk(
    original_linear: nn.Module,
    block_size: tuple,
    sparsity_ratio: float
) -> 'BlockSparseTopKLinear':
    blocksparse_linear = BlockSparseTopKLinear(
        original_linear.in_features,
        original_linear.out_features,
        block_size,
        sparsity_ratio,
        device=original_linear.weight.device,
        dtype=original_linear.weight.dtype
    )
    blocksparse_linear.weight = nn.Parameter(
        original_linear.weight.data.T.clone().contiguous(),
        requires_grad=False
    )
    if original_linear.bias is not None:
        blocksparse_linear.bias = nn.Parameter(
            original_linear.bias.data.clone(),
            requires_grad=False
        )
    return blocksparse_linear

def _replace_linear_with_blocksparse_thres(
    original_linear: nn.Module,
    block_size: tuple,
    thres: float
) -> 'BlockSparseThresLinear':
    blocksparse_linear = BlockSparseThresLinear(
        original_linear.in_features,
        original_linear.out_features,
        block_size,
        thres,
        device=original_linear.weight.device,
        dtype=original_linear.weight.dtype
    )
    blocksparse_linear.weight = nn.Parameter(
        original_linear.weight.data.T.clone().contiguous(),
        requires_grad=False
    )
    if original_linear.bias is not None:
        blocksparse_linear.bias = nn.Parameter(
            original_linear.bias.data.clone(),
            requires_grad=False
        )
    return blocksparse_linear

def patch_qwen2_topk_linear(model, block_size: tuple, sparsity_ratio: float):
    for layer in model.model.layers:
        assert isinstance(layer.mlp.gate_proj, nn.Linear)
        assert isinstance(layer.mlp.up_proj, nn.Linear)
        assert isinstance(layer.mlp.down_proj, nn.Linear)
        assert isinstance(layer.self_attn.q_proj, nn.Linear)
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
        assert isinstance(layer.self_attn.v_proj, nn.Linear)
        assert isinstance(layer.self_attn.o_proj, nn.Linear)
        layer.mlp.gate_proj = _replace_linear_with_blocksparse_topk(
            layer.mlp.gate_proj, block_size, sparsity_ratio
        )
        layer.mlp.up_proj = _replace_linear_with_blocksparse_topk(
            layer.mlp.up_proj, block_size, sparsity_ratio
        )
        layer.mlp.down_proj = _replace_linear_with_blocksparse_topk(
            layer.mlp.down_proj, block_size, sparsity_ratio
        )
        layer.self_attn.q_proj = _replace_linear_with_blocksparse_topk(
            layer.self_attn.q_proj, block_size, sparsity_ratio
        )
        layer.self_attn.k_proj = _replace_linear_with_blocksparse_topk(
            layer.self_attn.k_proj, block_size, sparsity_ratio
        )
        layer.self_attn.v_proj = _replace_linear_with_blocksparse_topk(
            layer.self_attn.v_proj, block_size, sparsity_ratio
        )
        layer.self_attn.o_proj = _replace_linear_with_blocksparse_topk(
            layer.self_attn.o_proj, block_size, sparsity_ratio
        )

def get_qwen2_thres_metric(model):
    results = []
    for layer in model.model.layers:
        results.append(
            {
                "gate_proj": layer.mlp.gate_proj.thres_metric.compute(),
                "up_proj": layer.mlp.up_proj.thres_metric.compute(),
                "down_proj": layer.mlp.down_proj.thres_metric.compute(),
                "q_proj": layer.self_attn.q_proj.thres_metric.compute(),
                "k_proj": layer.self_attn.k_proj.thres_metric.compute(),
                "v_proj": layer.self_attn.v_proj.thres_metric.compute(),
                "o_proj": layer.self_attn.o_proj.thres_metric.compute()
            }
        )
    return results

def patch_qwen2_thres_linear(model, block_size: tuple, thres: list):
    for idx, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp.gate_proj, nn.Linear)
        assert isinstance(layer.mlp.up_proj, nn.Linear)
        assert isinstance(layer.mlp.down_proj, nn.Linear)
        assert isinstance(layer.self_attn.q_proj, nn.Linear)
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
        assert isinstance(layer.self_attn.v_proj, nn.Linear)
        assert isinstance(layer.self_attn.o_proj, nn.Linear)
        layer.mlp.gate_proj = _replace_linear_with_blocksparse_thres(
            layer.mlp.gate_proj, block_size, thres[idx]["gate_proj"]
        )
        layer.mlp.up_proj = _replace_linear_with_blocksparse_thres(
            layer.mlp.up_proj, block_size, thres[idx]["up_proj"]
        )
        # layer.mlp.down_proj = _replace_linear_with_blocksparse_thres(
        #     layer.mlp.down_proj, block_size, thres[idx]["down_proj"]
        # )
        # layer.self_attn.q_proj = _replace_linear_with_blocksparse_thres(
        #     layer.self_attn.q_proj, block_size, thres[idx]["q_proj"]*0
        # )
        # layer.self_attn.k_proj = _replace_linear_with_blocksparse_thres(
        #     layer.self_attn.k_proj, block_size, thres[idx]["k_proj"]*0
        # )
        # layer.self_attn.v_proj = _replace_linear_with_blocksparse_thres(
        #     layer.self_attn.v_proj, block_size, thres[idx]["v_proj"]*0
        # )
        # layer.self_attn.o_proj = _replace_linear_with_blocksparse_thres(
        #     layer.self_attn.o_proj, block_size, thres[idx]["o_proj"]*0
        # )

def patch_qwen2(
        model, linear_type: Union[Literal["thres"], Literal["topk"]], 
        block_size: tuple, thres: float = None, sparsity_ratio: float = None
):
    assert linear_type in ["thres", "topk"], "linear_type must be 'thres' or 'topk'"
    if linear_type == "topk":
        assert thres is None, "thres must be None for topk"
        assert sparsity_ratio is not None, "sparsity_ratio must be specified for topk"
        patch_qwen2_topk_linear(model, block_size, sparsity_ratio)
    else:
        assert sparsity_ratio is None, "sparsity_ratio must be None for thres"
        assert thres is not None, "thres must be specified for thres"
        patch_qwen2_thres_linear(model, block_size, thres)