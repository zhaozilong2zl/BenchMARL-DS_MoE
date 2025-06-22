from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm

from tensordict import TensorDictBase
from torchrl.modules import MultiAgentMLP, MLP

from benchmarl.models.common import Model, ModelConfig


class MAPOCA(Model):
    """
    MA-POCA Model with self-attention for handling variable number of agents.
    implement according to https://arxiv.org/pdf/2111.05992.

    """

    def __init__(
        self,
        num_cells_e: Sequence[int],          # Encoder MLP
        num_cells_v: Sequence[int],          # Value MLP
        num_heads: int,                      # attention head
        embedding_dim: int,                  # attention embedding dim
        activation_class: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_features = sum([spec.shape[-1] for spec in self.input_spec.values(True, True)])
        self.output_features = self.output_leaf_spec.shape[-1]
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Encoder MLP：observation -> embedding, feed into self-attention
        self.encoder = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=embedding_dim,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            num_cells=num_cells_e,
            activation_class=activation_class,
        )

        # self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            device=self.device,
        )


        # critic value network
        self.value_network = MLP(
            in_features=embedding_dim,
            out_features=self.output_features,
            device=self.device,
            num_cells=num_cells_v,
            activation_class=activation_class,
        )

        self._params_counted = False  # flag: if get counting params yet
        self._init_params = -1  # flag: avoid counting params before instantiating

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        # parameter count
        if not self._params_counted:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            if self._init_params == total_params:
                print(f"\n--- Model Parameter Count (from {self.__class__.__name__} forward) ---")
                # add more information about the model
                model_info = f"Agent Group: {self.agent_group}, Model Index: {self.model_index}, Is Critic: {self.is_critic}"
                print(model_info)
                print(f"{self.__class__.__name__} Model Parameters: {total_params}")
                # print device
                try:
                    if any(p.requires_grad for p in self.parameters()):
                        print(
                            f"{self.__class__.__name__} Model Device: {next(p for p in self.parameters() if p.requires_grad).device}")
                    else:
                        print(f"{self.__class__.__name__} Model Device: N/A (No trainable parameters)")
                except StopIteration:
                    print(f"{self.__class__.__name__} Model Device: Device access failed (possibly no parameters?)")
                print("---------------------------------------------------------------\n")
                self._params_counted = True
            self._init_params = total_params

        input = torch.cat([tensordict.get(in_key) for in_key in self.in_keys], dim=-1)
        if input.dim() == 4:
            batch_size, max_steps, n_agents, _ = input.shape
            input = input.view(batch_size * max_steps, n_agents, -1)
        else:
            batch_size, n_agents, _ = input.shape
            max_steps = 1
            input = input.unsqueeze(1).view(batch_size * max_steps, n_agents, -1)

        active_mask = ~torch.all(input == 0, dim=-1)
        attn_mask = ~active_mask

        # 1. 计算掩码平均时使用的分母，加上一个很小的数防止除以零
        #    这个操作即使在没有活跃智能体的情况下也是安全的（分母为1e-6）
        num_active_agents = active_mask.sum(dim=-1, keepdim=True)
        safe_denominator = num_active_agents + 1e-6

        embeddings = self.encoder(input)

        attn_output, _ = self.attention(embeddings, embeddings, embeddings,
                                        key_padding_mask=attn_mask)

        # 残差连接
        residual_output = embeddings + attn_output  # 注意：这里使用 norm1_embeddings 而不是原始的 embeddings

        # 正确的掩码池化
        mask_unsqueezed = active_mask.unsqueeze(-1).float()
        pooled_output = (residual_output * mask_unsqueezed).sum(dim=1) / safe_denominator

        value = self.value_network(pooled_output)

        # 4. 再次掩码，确保终止状态（所有agent都不活跃）的价值输出严格为0
        # is_any_agent_active: [B,], bool
        is_any_agent_active = active_mask.any(dim=-1, keepdim=True)  # [B, 1]

        value = value * is_any_agent_active.float()

        if max_steps > 1:
            value = value.view(batch_size, max_steps, -1)
        else:
            value = value.view(batch_size, -1)

        tensordict.set(self.out_key, value)
        return tensordict


@dataclass
class MAPOCAModelConfig(ModelConfig):
    """Dataclass config for MAPOCAModel."""

    num_cells_e: Sequence[int] = MISSING
    num_cells_v: Sequence[int] = MISSING
    num_heads: int = MISSING
    embedding_dim: int = MISSING
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        return MAPOCA