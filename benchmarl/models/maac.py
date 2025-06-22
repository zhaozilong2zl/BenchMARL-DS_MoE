"""
Implementation of the MAAC critic model

"""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor
from torchrl.modules import MultiAgentMLP
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module used in the MAAC critic.

    Args:
        in_features (int): Input feature dimension for the queries, keys, and values, equals to the observation dim of agents.
        num_heads (int): Number of attention heads.
        device (str): Device to place tensors on (e.g., 'cuda:0' or 'cpu').

    """
    def __init__(self, in_features: int, num_heads: int, device: str):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = (in_features + num_heads - 1) // num_heads
        self.out_features = self.num_heads * self.head_dim
        self.query = nn.Linear(in_features, self.out_features).to(device)
        self.key = nn.Linear(in_features, self.out_features).to(device)
        self.value = nn.Linear(in_features, self.out_features).to(device)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))
        self.device = device

    def forward(self, embeddings: Tensor) -> Tensor:
        # embeddings: [batch_size, max_steps, n_agents, in_features] or [batch_size, n_agents, in_features]
        # Handle potential missing max_steps dimension
        if embeddings.dim() == 4:
            batch_size, max_steps, n_agents, in_features = embeddings.shape
            embeddings = embeddings.view(batch_size * max_steps, n_agents, in_features)
        else:
            batch_size, n_agents, in_features = embeddings.shape
            max_steps = 1
            embeddings = embeddings.view(batch_size * max_steps, n_agents, in_features)

        # Compute Q, K, V
        Q = self.query(embeddings)
        K = self.key(embeddings)
        V = self.value(embeddings)

        # Reshape and transpose for multi-head attention: [batch_size * max_steps, num_heads, n_agents, head_dim]
        Q = Q.view(batch_size * max_steps, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * max_steps, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * max_steps, n_agents, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # counting x_i excluding \alpha_i by using attn_mask, for equation below Eq. 5 in MAAC paper: x_i=\sum_{j \neq i} \alpha_j v_j
        self_mask = torch.eye(embeddings.shape[1], dtype=torch.bool, device=self.device)
        expanded_self_mask = self_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(expanded_self_mask, float('-inf'))

        # Compute attention weights and apply to values
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attention = torch.matmul(weights, V)
        attention = attention.transpose(1, 2).contiguous().view(batch_size * max_steps, n_agents, self.out_features)

        # recover max_steps dim
        attention = attention.view(batch_size, max_steps, n_agents, self.out_features)
        return attention


class Maac(Model):
    """
    MAAC critic model implementation, based on https://arxiv.org/abs/1810.02912

    The core idea of MAAC is to utilize attention over agent embeddings to select relevant information for estimating critics.

    To adapt the original MAAC structure for a MAPPO structure, this model outputs per-agent state values V_i(o_i).
    The parameters of the encoder and Q network are shared across all agents.
    With per-agent input features, the MAAC model computes per-agent state values.

    Args:
        num_cells_e (Sequence[int]): Number of cells for each hidden layer in the encoder MLP.
        num_cells_q (Sequence[int]): Number of cells for each hidden layer in the Q network MLP.
        num_heads (int): Number of attention heads in the MultiHeadAttention module.
        activation_class (Type[nn.Module]): Activation function class to use in MLPs.

    """

    def __init__(
        self,
        num_cells_e: list[int],
        num_cells_q: list[int],
        num_heads: int,
        activation_class: type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert self.input_has_agent_dim, "MAAC critic requires input with an agent dimension"

        # Total input features for each agent
        self.input_features = sum([spec.shape[-1] for spec in self.input_spec.values(True, True)])
        # Output feature dimension (typically 1 for value estimation)
        self.output_features = self.output_leaf_spec.shape[-1]

        # Encoder MLP: processes per-agent input features
        self.encoder = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=num_cells_e[-1],
            n_agents=self.n_agents,
            centralised=False,  # This MLP operates independently on each agent's input
            share_params=True,
            device=self.device,
            num_cells=num_cells_e,
            activation_class=activation_class,
        )

        # Attention module: computes context for each agent by attending to others' encoder outputs
        self.attention = MultiHeadAttention(
            in_features=num_cells_e[-1],
            num_heads=num_heads,
            device=self.device,
        )

        # Q Network MLP: takes concatenated encoder output and aggregated context as input
        self.q_network = MultiAgentMLP(
            n_agent_inputs=num_cells_e[-1] + self.attention.out_features,  # concatenate e_i and x_i according to Fig. 1
            n_agent_outputs=1,  # for PPO, output per-agent state value V_i(o_i) instead of Q_i(o_i, a_i)
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            num_cells=num_cells_q,
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
        # input: [batch_size, max_steps, n_agents, input_features] or [batch_size, n_agents, input_features]
        if input.dim() == 4:
            batch_size, max_steps, n_agents, _ = input.shape
        else:
            batch_size, n_agents, _ = input.shape
            max_steps = 1
            input = input.unsqueeze(1)  # add max_steps dim: [batch_size, 1, n_agents, input_features]

        # Encode per-agent input features
        e = self.encoder(input)

        # Compute aggregated context for each agent using attention (equation below Eq. 5 in MAAC paper)
        x = self.attention(e)

        # Concatenate encoder output (e_i) and aggregated context (x_i) as input for the Q network
        q_input = torch.cat([e, x], dim=-1)
        q = self.q_network(q_input)

        final_output = q
        if max_steps==1:  # if input dim without max_steps, or origin_input.dim()==3
            final_output = q.squeeze(1)
        tensordict.set(self.out_key, final_output)

        return tensordict


@dataclass
class MaacModelConfig(ModelConfig):
    num_cells_e: Sequence[int] = MISSING
    num_cells_q: Sequence[int] = MISSING
    num_heads: int = MISSING
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        return Maac
