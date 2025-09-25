# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBA2 model."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available
from .configuration_mamba import Mamba2Config


logger = logging.get_logger(__name__)


# if is_mamba_2_ssm_available():
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
#     from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
# else:
#     mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

# if is_causal_conv1d_available():
#     from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# else:
#     causal_conv1d_update, causal_conv1d_fn = None, None

# is_fast_path_available = all((
#     selective_state_update,
#     mamba_chunk_scan_combined,
#     mamba_split_conv1d_scan_combined,
#     causal_conv1d_fn,
#     causal_conv1d_update,
# ))
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from model.transformer.mamba.optimized_chunk_scan import mamba_chunk_scan_combined


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class Mamba2Cache:
    """
    Arguments:
        config: Mamba2Config
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config.
        n_groups: (`int`):
            Model's number of groups taken from the config - similar to tensor parallel in Transformer.
        state_size: (`int`):
            Model's SSM state size taken from config.
        num_heads: (`int`):
            The number of heads used in the linear attention / SSM.
        head_dim: (`int`):
            The respective dimension of the heads used in the linear attention / SSM.
        intermediate_size: (`int`):
            Model's intermediate_size based on (expand * hidden_dim) from config.
        conv_states: (`torch.Tensor`):
            A tensor of shape `[num_layers, batch_size, conv_kernel_size, intermediate_size + 2 * n_groups * state_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[num_layers, batch_size, num_heads, head_dim, state_size]` that holds ssm states.
    """

    def __init__(
        self, config: Mamba2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, :, 0].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
        )
        # self.B_proj = nn.Linear(self.head_dim, self.ssm_state_size, bias=False)
        # self.C_proj = nn.Linear(self.ssm_state_size, self.head_dim, bias=False)

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones((self.num_heads,)))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones((self.num_heads,)))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # getting projected states from cache if it exists
        do_inference = (cache_params is not None) and (cache_position is not None) and (cache_position[0] > 0)
        if do_inference:
            assert seq_len == 1, "Only supports inference for one token at a time."
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        projected_states = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(projected_states, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        hidden_states_B_C = hidden_states_B_C.transpose(1, 2)

        if do_inference:
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)
            padded_conv_states = cache_params.conv_states[self.layer_idx]




            hidden_states, B, C = torch.split(
                self.act(self.conv1d(padded_conv_states).transpose(1, 2)),
                [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size, ],
                dim=-1,
            )
            hidden_states = hidden_states.unflatten(-1, (self.num_heads, -1,))
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
            B = B.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)
            C = C.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)



        else:
            # 1D Convolution
            padded_conv_states = Fn.pad(hidden_states_B_C, (self.conv_kernel_size - 1, 0,))
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=padded_conv_states[..., -self.conv_kernel_size:], cache_init=True)






            from causal_conv1d import causal_conv1d_fn
            hidden_states_B_C = causal_conv1d_fn(
                x=hidden_states_B_C,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

            hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
                dim=-1,
            )

            hidden_states = hidden_states.unflatten(-1, (self.num_heads, -1,))
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
            B = B.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)
            C = C.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)



        # hidden_states, B, C = torch.split(
        #     self.act(self.conv1d(padded_conv_states).transpose(1, 2)),
        #     [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size,], dim=-1,
        # )
        # hidden_states = hidden_states.unflatten(-1, (self.num_heads, -1,))
        # hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        # B = B.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)
        # C = C.unflatten(-1, (self.n_groups, -1,)).repeat_interleave(self.num_heads // self.n_groups, dim=-2)






        A = -torch.exp(self.A_log.float())  # (nheads,)
        if do_inference:   
            dt = dt[:, 0, :, None].expand((-1, -1, self.head_dim,))
            dt_bias = self.dt_bias[:, None].expand((self.num_heads, self.head_dim,))
            A = A[:, None, None].expand((-1, self.head_dim, self.ssm_state_size,)).to(dtype=torch.float32)
            D = self.D[:, None].expand((self.num_heads, self.head_dim,))
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[:, 0],
                dt, A, B[:, 0], C[:, 0], D,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
        # if no cache is found, calling the kernel
        else:
            hidden_states, ssm_state = mamba_chunk_scan_combined(hidden_states, dt, self.dt_bias, A, B, C, self.D,
                                                                 dt_limit=self.time_step_limit, chunk_size=self.chunk_size)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)



        # dt = Fn.softplus(dt + self.dt_bias)
        # A = -torch.exp(self.A_log.float())  # (nheads,)
        
        # dA = torch.exp(dt * A[None, None, :])
        # dB = self.B_proj(dt[..., None] * hidden_states) * B

        # prod_states = torch.zeros((batch_size, seq_len, self.num_heads, self.ssm_state_size,))
        # ssm_state = cache_params.ssm_states[self.layer_idx] if do_inference else torch.zeros((batch_size, self.num_heads, self.ssm_state_size,))
        # for i in range(seq_len):
        #     ssm_state = dA[:, i, :, None] * ssm_state + dB[:, i, :, :]
        #     prod_states[:, i, :, :] = ssm_state * C[:, i, :, :]
        # hidden_states = self.C_proj(prod_states) + hidden_states * self.D[:, None]

        # if cache_params is not None:
        #     cache_params.ssm_states[self.layer_idx].copy_(ssm_state)



        # Multiply "gate" branch and apply extra normalization layer
        hidden_states = hidden_states.view((batch_size, seq_len, -1,))
        hidden_states = self.norm(hidden_states, gate)
        out = self.out_proj(hidden_states)

        return out


class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mamba2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mamba2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Mamba2PreTrainedModel(PreTrainedModel):
    config: Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, Mamba2Mixer):
            # S4D real initialization. These are not discretized!
            # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
            A = torch.arange(1, self.config.num_heads + 1)
            module.A_log.copy_(torch.log(A))
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
            module.D.data.fill_(1.0)

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

            nn.init.kaiming_uniform_(module.conv1d.weight, a=math.sqrt(5))
            if module.conv1d.bias is not None:
                if not getattr(module.conv1d.bias, "_no_reinit", False):
                    nn.init.zeros_(module.conv1d.bias)
            nn.init.kaiming_uniform_(module.out_proj.weight, a=math.sqrt(5))

            if self.config.rescale_prenorm_residual:
                # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
                #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
                #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
                #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
                #
                # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                p = module.out_proj.weight
                p /= math.sqrt(self.config.num_hidden_layers)

        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (Mamba2RMSNorm, MambaRMSNormGated)):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)


@dataclass
@auto_docstring(
    custom_intro="""
    Class for the MAMBA2 model outputs.
    """
)
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->Mamba2
class Mamba2Output(ModelOutput):
    r"""
    cache_params (`Mamba2Cache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for causal language model (or autoregressive) outputs.
    """
)
# Copied from transformers.models.mamba.modeling_mamba.MambaCausalLMOutput with Mamba->Mamba2
class Mamba2CausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`Mamba2Cache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class Mamba2Model(Mamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Mamba2Output]:
        r"""
        cache_params (`Mamba2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The MAMBA2 Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    """
)
class Mamba2ForCausalLM(Mamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        model_inputs = {"input_ids": input_ids.contiguous()}
        if use_cache and cache_params is None:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            cache_position = torch.arange(0, self.backbone.config.conv_kernel, device=input_ids.device)
            if inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}
                max_batch_size = inputs_embeds.size(0)
            else:
                max_batch_size = input_ids.size(0)
            cache_params = Mamba2Cache(self.backbone.config, max_batch_size, device=self.device, dtype=self.dtype)

        if use_cache and cache_position[0] > 0:
            model_inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1).contiguous()
            attention_mask = None

        if not use_cache and inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation and loss_function
    ) -> Union[tuple, Mamba2CausalLMOutput]:
        r"""
        cache_params (`Mamba2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba2_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + mamba2_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba2_outputs.cache_params,
            hidden_states=mamba2_outputs.hidden_states,
        )


__all__ = ["Mamba2ForCausalLM", "Mamba2Model", "Mamba2PreTrainedModel"]
