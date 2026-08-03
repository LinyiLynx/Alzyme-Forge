from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainConfig


class Lookahead(Optimizer):
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group) -> None:
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        fast_state = self.optimizer.state_dict()
        slow_state = {(id(key) if isinstance(key, torch.Tensor) else key): value for key, value in self.state.items()}
        return {
            "fast_state": fast_state["state"],
            "slow_state": slow_state,
            "param_groups": fast_state["param_groups"],
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {"state": state_dict["slow_state"], "param_groups": state_dict["param_groups"]}
        fast_state_dict = {"state": state_dict["fast_state"], "param_groups": state_dict["param_groups"]}
        super().load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state


def parameter_groups(model: torch.nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> Optimizer:
    groups = parameter_groups(model, config.weight_decay)
    if config.optimizer == "AdamW":
        return AdamW(groups, lr=config.lr)
    if config.optimizer == "RAdam+Lookahead":
        inner = torch.optim.RAdam(groups, lr=config.lr)
        return Lookahead(inner, k=5, alpha=0.5)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(optimizer: Optimizer, config: TrainConfig, steps_per_epoch: int):
    if config.scheduler == "none":
        return None
    if config.scheduler != "cosine":
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    total_steps = max(config.epochs * steps_per_epoch, 1)
    warmup_steps = config.warmup_epochs * steps_per_epoch

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=schedule)
