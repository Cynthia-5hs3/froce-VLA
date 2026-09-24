"""Local low-rank updates for the PI0.5 language encoder and action expert."""

import math

import torch
from torch import nn
import torch.nn.functional as functional


class LoRALinear(nn.Module):
    def __init__(self, source, rank, alpha):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.weight = source.weight
        self.bias = source.bias
        self.in_features, self.out_features = source.in_features, source.out_features
        self.scaling = alpha / rank
        self.lora_down = nn.Parameter(source.weight.new_empty(rank, source.in_features))
        self.lora_up = nn.Parameter(source.weight.new_zeros(source.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))

    def forward(self, value):
        base = functional.linear(value, self.weight, self.bias)
        delta = functional.linear(functional.linear(value, self.lora_down), self.lora_up)
        return base + delta * self.scaling


def install_lora(model, rank, alpha, expert_rank=None, expert_alpha=None):
    roots = (
        "paligemma_with_expert.paligemma.model.language_model.",
        "paligemma_with_expert.gemma_expert.model.",
    )
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    counts = {root: 0 for root in roots}
    for name, module in list(model.named_modules()):
        root = next((prefix for prefix in roots if name.startswith(prefix)), None)
        if root is None or name.rsplit(".", 1)[-1] not in targets:
            continue
        selected_rank = expert_rank if root == roots[1] and expert_rank is not None else rank
        selected_alpha = expert_alpha if root == roots[1] and expert_alpha is not None else alpha
        if isinstance(module, LoRALinear):
            if module.lora_down.shape[0] != selected_rank or module.scaling != selected_alpha / selected_rank:
                raise ValueError("Existing LoRA configuration differs")
        elif isinstance(module, nn.Linear):
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, LoRALinear(module, selected_rank, selected_alpha))
        else:
            raise ValueError(f"Unsupported LoRA projection: {name}")
        counts[root] += 1
    if not all(counts.values()):
        raise ValueError(f"LoRA must cover both encoder and decoder: {counts}")
    return counts
