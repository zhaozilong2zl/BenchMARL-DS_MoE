from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn, Tensor
from torchrl.modules import MLP
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig

import os
import csv
from datetime import datetime

class MoEnet(nn.Module):
    def __init__(self,in_features, out_features, n_agents, num_experts=2, num_cells=[128, 128, 128, 128],
                 activation_class=nn.ReLU, device="cuda",
                 log_file_prefix="/home/l/Proj_folder/research/TNNLS_ds_moe/experiments/gate_test/moe_log",log_frequency=10,
                 use_quadratic_gate=True):
        super().__init__()
        self.num_experts = num_experts
        self.device = device

        # self.input_layernorm = nn.LayerNorm(in_features, device=device)
        
        # 2-Experts structure
        self.experts = nn.ModuleList([
            MLP(in_features=in_features,
                out_features=out_features,
                num_cells=[num_cells[0], num_cells[1]],
                activation_class=activation_class,
                device=device),
            MLP(in_features=in_features,
                out_features=out_features,
                num_cells=[num_cells[2], num_cells[3]],
                activation_class=activation_class,
                device=device)
        ])

        # 门控网络 (Gate)
        self.use_quadratic_gate = use_quadratic_gate
        self.in_features = in_features
        if self.use_quadratic_gate:
            # 每个专家 i 对应一个 A_i (in_features x in_features) 和 c_i (scalar)
            # 使用 ParameterList 来正确注册参数
            self.quadratic_A_params = nn.ParameterList()
            for _ in range(num_experts):
                # 在创建时就放到正确的设备上
                param = nn.Parameter(torch.randn(self.in_features, self.in_features, device=self.device))
                nn.init.xavier_uniform_(param)  # 初始化可以之后进行 xavier
                self.quadratic_A_params.append(param)

            self.quadratic_c_params = nn.ParameterList()
            for _ in range(num_experts):
                param = nn.Parameter(torch.randn(1, device=self.device))
                nn.init.zeros_(param)  # 初始化
                self.quadratic_c_params.append(param)

        else:  # 使用原始的线性门控
            self.gate = nn.Linear(self.in_features, num_experts).to(device)


    def forward(self, x):

        # x:  [batch_size, max_steps, agents, feature] or [batch_size * max_steps, agents, feature]
        if x.dim() > 4:  # if have set(group) dim, [batch_size * max_steps, agents, set, feature]
            x_gate = x.mean(dim=-2)
        else:
            x_gate = x

        # x_gate = self.input_layernorm(x_gate)
        if self.use_quadratic_gate:
            # x_gate 形状: [..., self.in_features_gate]
            expert_scores = []
            for i in range(self.num_experts):
                A_i = self.quadratic_A_params[i]  # [F, F]
                A_i_sym = 0.5 * (A_i + A_i.T)
                c_i = self.quadratic_c_params[i]  # [1]

                # 计算 x^T A_i x
                # (x_gate @ A_i) 结果是 [..., F]
                # 然后 (x_gate @ A_i) * x_gate 也是 [..., F]
                # 最后 sum over F -> [...,]
                quadratic_term = torch.sum((x_gate @ A_i_sym) * x_gate, dim=-1)  # 结果 [...,]
                score = quadratic_term + c_i.squeeze()  # 确保 c_i 是标量或能广播
                expert_scores.append(score)

            stacked_scores = torch.stack(expert_scores, dim=-1)  # [..., num_experts]
            gates = torch.softmax(stacked_scores, dim=-1)  # [..., num_experts]
        else:
            # 原始线性门控
            gates = torch.softmax(self.gate(x_gate), dim=-1)  # [..., num_experts]

        expert_outputs = torch.stack([e(x_gate) for e in self.experts], dim=-1)

        if x.dim() > 4:
            gates = gates.unsqueeze(-2).expand(-1, -1, x.shape[-2], -1)  # [batch_size, n_agents, S, num_experts]
        return torch.sum(gates.unsqueeze(-2) * expert_outputs, dim=-1)



class MoE(Model):
    """
    Base on Benchmarl DeepSets implementation, change `\phi` MLP to MoEnet.

    Args:
        aggr (str): The aggregation strategy to use in the Deepsets model.
        local_nn_num_cells (Sequence[int]): number of cells of every layer in between the input and output in the :math:`\phi` MLP.
        local_nn_activation_class (Type[nn.Module]): activation class to be used in the :math:`\phi` MLP.
        out_features_local_nn (int): output features of the :math:`\phi` MLP.
        global_nn_num_cells (Sequence[int]): number of cells of every layer in between the input and output in the :math:`\rho` MLP.
        global_nn_activation_class (Type[nn.Module]): activation class to be used in the :math:`\rho` MLP.

    """

    def __init__(
        self,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        num_experts: int = 2,
        use_quadratic_gate: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.aggr = aggr
        self.local_nn_num_cells = local_nn_num_cells
        self.local_nn_activation_class = local_nn_activation_class
        self.global_nn_num_cells = global_nn_num_cells
        self.global_nn_activation_class = global_nn_activation_class
        self.out_features_local_nn = out_features_local_nn
        self.num_experts = num_experts
        self.use_quadratic_gate = use_quadratic_gate

        self.input_local_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_local]
        )
        self.input_local_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_local]
        )
        self.input_global_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_global]
        )
        self.input_global_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_global]
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_local_set_features > 0:  # Need local deepsets
            self.local_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=self.input_local_set_features,
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_local_tensor_features,
                        out_features=(
                            self.output_features
                            if not self.centralised
                            else self.out_features_local_nn
                        ),
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        if self.centralised:  # Need global deepsets
            self.global_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=(
                            self.input_global_set_features
                            # self.input_global_set_features*2  ############################################################################################
                            if self.input_local_set_features == 0
                            else self.out_features_local_nn
                        ),
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_global_tensor_features,
                        out_features=self.output_features,
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        self._params_counted = False  # flag: if get counting params yet
        self._init_params = -1  # flag: avoid counting params before instantiating

    def _make_deepsets_net(
        self,
        in_features: int,
        out_features: int,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        in_fetures_global_nn: int,
    ) -> _DeepsetsNet:
        local_nn = MoEnet(
            in_features=in_features,
            out_features=out_features_local_nn,
            n_agents=self.n_agents,
            num_experts=self.num_experts,
            use_quadratic_gate=self.use_quadratic_gate,
            num_cells=local_nn_num_cells,
            activation_class=local_nn_activation_class,
            device=self.device,
        )
        global_nn = MLP(
            in_features=in_fetures_global_nn,
            out_features=out_features,
            num_cells=global_nn_num_cells,
            activation_class=global_nn_activation_class,
            device=self.device,
        )
        return _DeepsetsNet(local_nn, global_nn, aggr=aggr)

    def _perform_checks(self):
        super()._perform_checks()

        input_shape_tensor_local = None
        self.tensor_in_keys_local = []
        input_shape_set_local = None
        self.set_in_keys_local = []

        input_shape_tensor_global = None
        self.tensor_in_keys_global = []
        input_shape_set_global = None
        self.set_in_keys_global = []

        error_invalid_input = ValueError(
            f"DeepSet set inputs should all have the same shape up to the last dimension, got {self.input_spec}"
        )

        for input_key, input_spec in self.input_spec.items(True, True):
            if self.input_has_agent_dim and len(input_spec.shape) == 3:
                self.set_in_keys_local.append(input_key)
                if input_shape_set_local is None:
                    input_shape_set_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_local:
                    raise error_invalid_input
            elif self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.tensor_in_keys_local.append(input_key)
                if input_shape_tensor_local is None:
                    input_shape_tensor_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_local:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.set_in_keys_global.append(input_key)
                if input_shape_set_global is None:
                    input_shape_set_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_global:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 1:
                self.tensor_in_keys_global.append(input_key)
                if input_shape_tensor_global is None:
                    input_shape_tensor_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_global:
                    raise error_invalid_input
            else:
                raise ValueError(
                    f"DeepSets input value {input_key} from {self.input_spec} has an invalid shape"
                )

        # Centralized model not needing any local deepsets
        if (
            self.centralised
            and not len(self.set_in_keys_local)
            and self.input_has_agent_dim
        ):
            self.set_in_keys_global = self.tensor_in_keys_local
            input_shape_set_global = input_shape_tensor_local
            self.tensor_in_keys_local = []

        if (not self.centralised and not len(self.set_in_keys_local)) or (
            self.centralised
            and not self.input_has_agent_dim
            and not len(self.set_in_keys_global)
        ):
            raise ValueError("DeepSets found no set inputs, maybe use an MLP?")

        if len(self.set_in_keys_local) and input_shape_set_local[-2] != self.n_agents:
            raise ValueError()
        if (
            len(self.tensor_in_keys_local)
            and input_shape_tensor_local[-1] != self.n_agents
        ):
            raise ValueError()
        if (
            len(self.set_in_keys_global)
            and self.input_has_agent_dim
            and input_shape_set_global[-1] != self.n_agents
        ):
            raise ValueError()

        if (
            self.output_has_agent_dim
            and (
                self.output_leaf_spec.shape[-2] != self.n_agents
                or len(self.output_leaf_spec.shape) != 2
            )
        ) or (not self.output_has_agent_dim and len(self.output_leaf_spec.shape) != 1):
            raise ValueError()

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
        
        if len(self.set_in_keys_local):
            # Local deep sets
            input_local_sets = torch.cat(
                [tensordict.get(in_key) for in_key in self.set_in_keys_local], dim=-1
            )
            input_local_tensors = None
            if len(self.tensor_in_keys_local):
                input_local_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_local],
                    dim=-1,
                )
            if self.share_params:
                local_output = self.local_deepsets[0](
                    input_local_sets, input_local_tensors
                )
            else:
                local_output = torch.stack(
                    [
                        net(input_local_sets, input_local_tensors)[..., i, :]
                        for i, net in enumerate(self.local_deepsets)
                    ],
                    dim=-2,
                )
        else:
            local_output = None

        if self.centralised:
            if local_output is None:
                # gather local output
                local_output = torch.cat(
                    [tensordict.get(in_key) for in_key in self.set_in_keys_global],
                    dim=-1,
                )

            # # --- 步骤 1: 计算全局平均场向量 ---
            # # 我们对“智能体”维度 (dim=-2) 求平均值。
            # # keepdim=True 是一个至关重要的技巧，它让输出的维度保持为 [Batch, 1, Features]
            # # 而不是 [Batch, Features]。这使得后续的广播操作极其方便。
            # mean_context = torch.mean(local_output, dim=-2, keepdim=True)
            # 
            # # --- 步骤 2: 将平均场广播给每一个智能体 ---
            # # expand_as() 是一个零拷贝、高效率的操作。
            # # 它将 [Batch, 1, Features] 的 mean_context “复制”N_Agents份，
            # # 使其形状变为与 input_local_sets 相同的 [Batch, N_Agents, Features]。
            # # 这样，每个智能体都拥有了同一个全局上下文的拷贝。
            # tiled_mean_context = mean_context.expand_as(local_output)
            # 
            # # --- 步骤 3: 创建包含上下文的“增强输入” ---
            # # 我们在特征维度 (dim=-1)上，将智能体自身的观测和全局上下文拼接起来。
            # 
            # enriched_input = torch.cat([local_output, tiled_mean_context], dim=-1)


            # Global deepsets
            input_global_tensors = None
            if len(self.tensor_in_keys_global):
                input_global_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_global],
                    dim=-1,
                )
            if self.share_params:
                global_output = self.global_deepsets[0](
                    local_output, input_global_tensors
                )
            # if self.share_params:
            #     global_output = self.global_deepsets[0](
            #         enriched_input, input_global_tensors
            #     )
            else:
                global_output = torch.stack(
                    [
                        net(local_output, input_global_tensors)
                        for i, net in enumerate(self.global_deepsets)
                    ],
                    dim=-2,
                )
            tensordict.set(self.out_key, global_output)
        else:
            tensordict.set(self.out_key, local_output)

        return tensordict


class _DeepsetsNet(nn.Module):
    """https://arxiv.org/abs/1703.06114"""

    def __init__(
        self,
        local_nn: torch.nn.Module,
        global_nn: torch.nn.Module,
        set_dim: int = -2,
        aggr: str = "sum",
    ):
        super().__init__()
        self.aggr = aggr
        self.set_dim = set_dim
        self.local_nn = local_nn
        self.global_nn = global_nn

        # # ---- Norm 1: For the Local/Phi Block ----
        # self.norm_local = nn.LayerNorm(local_nn.in_features, device=local_nn.device)
        # 
        # # ---- Norm 2: For the Global/Rho Block ----
        # self.norm_global = nn.LayerNorm(global_nn.in_features, device=local_nn.device)

    def forward(self, x: Tensor, extra_global_input: Optional[Tensor]) -> Tensor:

        # # --- Part 1: Local Processing Block (Phi function) ---
        # # The first complete Pre-LN residual block
        # # x_after_local = x + self.local_nn(self.norm_local(x))
        # # x_after_local = self.local_nn(self.norm_local(x))
        # x_after_local = self.local_nn(x)
        # 
        # # --- Part 2: Aggregation ---
        # aggregated = self.reduce(x_after_local, dim=self.set_dim, aggr=self.aggr)
        # 
        # # --- Part 3: Global Processing Block (Rho function) ---
        # # Note: A residual connection around the final MLP is less common but possible.
        # # Here, simply normalizing the input to global_nn is the most crucial step.
        # if extra_global_input is not None:
        #     global_input = torch.cat([aggregated, extra_global_input], dim=-1)
        # else:
        #     global_input = aggregated
        # 
        # # We apply LayerNorm before the final processing layer.
        # final_output = self.global_nn(global_input)
        # 
        # return final_output

        # # 1. 归一化输入
        # normalized_x = self.layernorm_before_local(x)
        # # 2. 通过 Sublayer (即你的 MoEnet)
        # transformed_x = self.local_nn(normalized_x)
        # # 3. 添加残差连接 (高速公路)
        # x_residual = x + transformed_x
        # # 使用处理后的 x_residual 进行后续操作
        # aggregated_x = self.reduce(x_residual, dim=self.set_dim, aggr=self.aggr)
        # if extra_global_input is not None:
        #     final_input_for_global = torch.cat([aggregated_x, extra_global_input], dim=-1)
        # else:
        #     final_input_for_global = aggregated_x
        # output = self.global_nn(final_input_for_global)
        # return output


        x = self.local_nn(x)
        x = self.reduce(x, dim=self.set_dim, aggr=self.aggr)
        if extra_global_input is not None:
            x = torch.cat([x, extra_global_input], dim=-1)
        x = self.global_nn(x)
        return x

    @staticmethod
    def reduce(x: Tensor, dim: int, aggr: str) -> Tensor:
        if aggr == "sum" or aggr == "add":
            return torch.sum(x, dim=dim)
        elif aggr == "mean":
            return torch.mean(x, dim=dim)
        elif aggr == "max":
            return torch.max(x, dim=dim)[0]
        elif aggr == "min":
            return torch.min(x, dim=dim)[0]
        elif aggr == "mul":
            return torch.prod(x, dim=dim)


@dataclass
class MoEConfig(ModelConfig):
    aggr: str = MISSING
    out_features_local_nn: int = MISSING
    local_nn_num_cells: Sequence[int] = MISSING
    local_nn_activation_class: Type[nn.Module] = MISSING
    global_nn_num_cells: Sequence[int] = MISSING
    global_nn_activation_class: Type[nn.Module] = MISSING
    num_experts: int = MISSING
    use_quadratic_gate: bool = MISSING

    @staticmethod
    def associated_class():
        return MoE
