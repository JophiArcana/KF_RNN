import itertools
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.static import ModelPair
from kf_rnn.model.convolutional.base import ConvolutionalPredictor
from kf_rnn.model.least_squares_predictor import LeastSquaresPredictor
from kf_rnn.infrastructure.config.schema import TrainConfig, controller_dims


class CnnPredictor(ConvolutionalPredictor):
    @dataclass
    class Config(ConvolutionalPredictor.Config):
        ir_length: int = 1

    def __init__(self, modelArgs: "CnnPredictor.Config"):
        ConvolutionalPredictor.__init__(self, modelArgs)
        self.ir_length = modelArgs.ir_length

        self.input_IR = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((v, self.ir_length, self.O_D)))                         # [? x R x O_D]
            for k, v in controller_dims(self.problem_shape).items()
        })
        self.observation_IR = nn.Parameter(torch.zeros((self.O_D, self.ir_length, self.O_D)))   # [O_D x R x O_D]


class CnnLeastSquaresPredictor(CnnPredictor, LeastSquaresPredictor):
    def __init__(self, modelArgs: "CnnLeastSquaresPredictor.Config"):
        CnnPredictor.__init__(self, modelArgs)
        LeastSquaresPredictor.__init__(self, modelArgs)

    def training_recipe(self) -> Sequence[str]:
        return ["online_least_squares"]

    @classmethod
    def terminate_least_squares_online(
            cls,
            THP: TrainConfig,
            exclusive: SimpleNamespace,
            model_pair: ModelPair,
            cache: SimpleNamespace,
    ) -> bool:
        # cache.index / cache.L are created by the first train_least_squares_online call,
        # so do not terminate before that bootstrap has run.
        if not hasattr(cache, "index"):
            return False
        return cache.index >= cache.L

    def train_least_squares_online(
            self,
            THP: TrainConfig,
            exclusive: SimpleNamespace,
            model_pair: ModelPair,
            cache: SimpleNamespace,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # SECTION: Setup the index dataloader, optimizer, and scheduler before running iterative training
        if not hasattr(cache, "index"):
            # TODO: Set up the dataset index sampler
            dataset: TensorDict = exclusive.train_info.dataset
            dataset = dataset.flatten(2, -2)
            cache.bsz = dataset.shape[:2]

            cache.B, cache.L = dataset.shape[-2:]
            cache.index = 0

            actions, observations = dataset["controller"], dataset["environment"]["observation"]
            cache.observations = observations                                                   # float: [NE... x B x L x O_D]
            cache.padded_observations = torch.cat([
                torch.zeros_like(observations[..., :1, :]),
                observations[..., :-1, :],
                torch.zeros_like(observations[..., -1:, :]),
            ], dim=-2)                                                                          # float: [NE... x B x (L + 1) x O_D]

            cache.ac_names = list(actions.keys())
            padded_actions = TensorDict.cat([actions, actions[..., -1:].apply(torch.zeros_like),], dim=-1)
            cache.padded_concatenated_data = torch.cat([
                *map(padded_actions.__getitem__, cache.ac_names),
                cache.padded_observations,
            ], dim=-1)                                                                          # float: [NE... x B x L x F]
            cache.cum_lengths = [0] + np.cumsum([*map(controller_dims(self.problem_shape).__getitem__, cache.ac_names), self.O_D]).tolist()

            cache.rf = self.ir_length * cache.cum_lengths[-1]
            cache.X = torch.zeros((*cache.bsz, 0, cache.rf,))                                   # float: [NE... x 0 x RF]
            cache.y = torch.zeros((*cache.bsz, 0, self.O_D,))                                   # float: [NE... x 0 x O_D]
            cache.XTX = torch.zeros((*cache.bsz, cache.rf, cache.rf,))                          # float: [NE... x RF x RF]
            cache.XTy = torch.zeros((*cache.bsz, cache.rf, self.O_D,))                          # float: [NE... x RF x O_D]
            cache.yTy = torch.zeros((*cache.bsz, self.O_D, self.O_D,))                          # float: [NE... x O_D x O_D]

        old_index = cache.index
        cache.index = min(cache.index + THP.sampling.batch_size, cache.L)

        chunk_indices = torch.arange(old_index, cache.index)                                    # int: [l]
        indices = (chunk_indices[:, None] - torch.arange(self.ir_length)).clamp_min(-1)         # int: [l x R]

        X = cache.padded_concatenated_data[..., indices, :]                                     # float: [NE... x B x l x R x F]
        y = cache.observations[..., chunk_indices, :]                                           # float: [NE... x B x l x O_D]
        flattened_X = einops.rearrange(X, "... b l r f -> ... (b l) (r f)")                     # float: [NE... x Bl x RF]
        flattened_y = einops.rearrange(y, "... b l d -> ... (b l) d")                           # float: [NE... x Bl x O_D]

        cache.XTX = cache.XTX + (flattened_X.mT @ flattened_X)
        cache.XTy = cache.XTy + (flattened_X.mT @ flattened_y)
        cache.yTy = cache.yTy + (flattened_y.mT @ flattened_y)
        try:
            XTX_lI_inv = torch.inverse(cache.XTX + self.ridge * torch.eye(cache.rf))            # float: [NE... x RF x RF]
            flattened_w = XTX_lI_inv @ cache.XTy                                                # float: [NE... x RF x O_D]
            error = eu.batch_trace(
                cache.yTy + cache.XTy.mT @ (XTX_lI_inv @ cache.XTX @ XTX_lI_inv - 2 * XTX_lI_inv) @ cache.XTy
            ) / (cache.B * cache.L)                                                             # float: [NE...]
        except torch.linalg.LinAlgError:
            # ``XTX`` is singular (not yet enough samples to identify the filter);
            # fall back to a pseudo-inverse on the accumulated design matrix.
            cache.X = torch.cat((cache.X, flattened_X,), dim=-2)
            cache.y = torch.cat((cache.y, flattened_y,), dim=-2)
            flattened_w = torch.linalg.pinv(cache.X) @ cache.y                                  # float: [NE... x RF x O_D]
            error = torch.zeros(cache.bsz)                                                      # float: [NE...]

        w = einops.rearrange(flattened_w, "... (r f) d -> ... f r d", r=self.ir_length)         # float: [NE... x F x R x O_D]
        weights = [w[..., lo:hi, :, :] for lo, hi in itertools.pairwise(cache.cum_lengths)]
        weights_dict = {"input_IR": dict(zip(cache.ac_names, weights[:-1])), "observation_IR": weights[-1],}
        for k, v in model_pair[1].items(include_nested=True, leaves_only=True):
            model_pair[1][k] = eu.rgetitem(weights_dict, k if isinstance(k, str) else ".".join(k)).expand_as(v)

        return error[None], {}


class CnnLeastSquaresPretrainPredictor(CnnLeastSquaresPredictor):
    def training_recipe(self) -> Sequence[str]:
        return [*CnnLeastSquaresPredictor.training_recipe(self), "sgd"]


class CnnAnalyticalPredictor(CnnPredictor):
    def analytical_initialization(self, exclusive: SimpleNamespace) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return eu.multi_vmap(
            self._analytical_initialization, 2,
            randomness="different",
        )(exclusive.train_info.systems.td().to_dict())

    def training_recipe(self) -> Sequence[str]:
        return ["analytical_init"]

    def _analytical_initialization(self, system_state_dict: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
        F, H, K = map(system_state_dict["environment"].__getitem__, ("F", "H", "K"))
        B = system_state_dict["environment"].get("B", {})
        S_D = F.shape[0]

        powers = eu.pow_series(F @ (torch.eye(S_D) - K @ H), self.ir_length)                 # [R x S_D x S_D]
        return {
            "input_IR": {
                k: (H @ powers @ B[k]).permute(2, 0, 1)                                         # [I_D x R x O_D]
                for k in controller_dims(self.problem_shape)
            },
            "observation_IR": (H @ powers @ (F @ K)).permute(2, 0, 1)                           # [O_D x R x O_D]
        }, torch.full((), torch.nan)

    def __init__(self, modelArgs: "CnnAnalyticalPredictor.Config"):
        CnnPredictor.__init__(self, modelArgs)
        self._initialization_error: torch.Tensor = None


class CnnAnalyticalLeastSquaresPredictor(CnnPredictor):
    def newton_initialization(self, stacked_modules: TensorDict, exclusive: SimpleNamespace) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."

        # Remove parameters that do not require gradients before flattening
        _fir_dict = OrderedDict([(k, v) for k, v in stacked_modules.items() if v.requires_grad])            # [N x E x F...]
        _flattened_fir_dict = torch.cat([v.flatten(2, -1) for v in _fir_dict.values()], dim=-1)             # [N x E x F]
        cum_lengths = [0] + np.cumsum([np.prod(v.shape[2:]) for v in _fir_dict.values()]).tolist()

        optimizer = optim.SGD(_fir_dict.values(), lr=0.0)
        def zero_grad_fir_dict() -> None:
            optimizer.zero_grad()
        zero_grad_fir_dict()

        L = ConvolutionalPredictor.analytical_error(stacked_modules, exclusive.train_info.systems.td())["environment", "observation"]    # [N x E]
        L.sum().backward(create_graph=True, retain_graph=True)
        _flattened_fir_grad_dict = torch.cat([v.grad.flatten(2, -1) for v in _fir_dict.values()], dim=-1)   # [N x E x F]
        zero_grad_fir_dict()

        H = []
        for f in range(_flattened_fir_dict.shape[-1]):
            _flattened_fir_grad_dict[:, :, f].sum().backward(create_graph=True, retain_graph=True)          # [N x E]
            H.append(torch.cat([v.grad.flatten(2, -1) for v in _fir_dict.values()], dim=-1))                # [N x E x F]
            zero_grad_fir_dict()
        H = torch.stack(H, dim=-2)                                                                          # [N x E x F x F]
        assert torch.allclose(H, H.mT), f"Computed Hessian must be symmetric up to numerical precision but got {H}."

        _flattened_newton_step = (torch.inverse(H) @ _flattened_fir_grad_dict.unsqueeze(-1)).squeeze(-1)    # [N x E x F]
        _newton_step = {
            k: _flattened_newton_step[:, :, cum_lengths[i]:cum_lengths[i + 1]].view_as(v)
            for i, (k, v) in enumerate(_fir_dict.items())
        }

        return TensorDict.from_dict({
            (*k.split("."),): v - _newton_step.get(k, 0.)
            for k, v in stacked_modules.items(include_nested=True, leaves_only=True)
        }, batch_size=stacked_modules.shape), torch.full((), torch.nan)

    def training_recipe(self) -> Sequence[str]:
        return ["newton_init"]


class CnnLeastSquaresRandomStepPredictor(CnnLeastSquaresPretrainPredictor):
    def training_recipe(self) -> Sequence[str]:
        return [*CnnLeastSquaresPretrainPredictor.training_recipe(self), "random_step"]


class CnnLeastSquaresNegationPredictor(CnnLeastSquaresPretrainPredictor):
    def training_recipe(self) -> Sequence[str]:
        return [*CnnLeastSquaresPretrainPredictor.training_recipe(self), "negation"]




