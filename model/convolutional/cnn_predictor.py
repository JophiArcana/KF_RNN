import itertools
from argparse import Namespace
from collections import OrderedDict
from typing import *

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.training import TrainFunc
from model.base import Predictor
from model.convolutional.base import ConvolutionalPredictor
from model.least_squares_predictor import LeastSquaresPredictor


class CnnPredictor(ConvolutionalPredictor):
    def __init__(self, modelArgs: Namespace):
        ConvolutionalPredictor.__init__(self, modelArgs)
        self.ir_length = modelArgs.ir_length

        self.input_IR = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((v, self.ir_length, self.O_D)))                         # [? x R x O_D]
            for k, v in vars(self.problem_shape.controller).items()
        })
        self.observation_IR = nn.Parameter(torch.zeros((self.O_D, self.ir_length, self.O_D)))   # [O_D x R x O_D]


class CnnLeastSquaresPredictor(CnnPredictor, LeastSquaresPredictor):
    def __init__(self, modelArgs: Namespace):
        CnnPredictor.__init__(self, modelArgs)
        LeastSquaresPredictor.__init__(self, modelArgs)

    def train_func_list(self, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return self.train_least_squares_online,

    def train_least_squares_online(
        self,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        def terminate_condition() -> bool:
            return cache.index >= cache.L

        # SECTION: Setup the index dataloader, optimizer, and scheduler before running iterative training
        if not hasattr(cache, "index"):
            # TODO: Set up the dataset index sampler
            dataset: TensorDict[str, torch.tensor] = exclusive.train_info.dataset.obj
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
            cache.cum_lengths = [0] + np.cumsum([*map(vars(self.problem_shape.controller).__getitem__, cache.ac_names), self.O_D]).tolist()

            cache.rf = self.ir_length * cache.cum_lengths[-1]
            cache.X = torch.zeros((*cache.bsz, 0, cache.rf,))                                   # float: [NE... x 0 x RF]
            cache.y = torch.zeros((*cache.bsz, 0, self.O_D,))                                   # float: [NE... x 0 x O_D]
            cache.XTX = torch.zeros((*cache.bsz, cache.rf, cache.rf,))                          # float: [NE... x RF x RF]
            cache.XTy = torch.zeros((*cache.bsz, cache.rf, self.O_D,))                          # float: [NE... x RF x O_D]
            cache.yTy = torch.zeros((*cache.bsz, self.O_D, self.O_D,))                          # float: [NE... x O_D x O_D]

        increment = 1
        old_index = cache.index
        cache.index = min(cache.index + increment, cache.L)

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
            error = utils.batch_trace(
                cache.yTy + cache.XTy.mT @ (XTX_lI_inv @ cache.XTX @ XTX_lI_inv - 2 * XTX_lI_inv) @ cache.XTy
            ) / (cache.B * cache.L)                                                             # float: [NE...]
        except Exception:
            cache.X = torch.cat((cache.X, flattened_X,), dim=-2)
            cache.y = torch.cat((cache.y, flattened_y,), dim=-2)
            flattened_w = torch.linalg.pinv(cache.X) @ cache.y                                  # float: [NE... x RF x O_D]
            error = torch.zeros(cache.bsz)                                                      # float: [NE...]

        w = einops.rearrange(flattened_w, "... (r f) d -> ... f r d", r=self.ir_length)         # float: [NE... x F x R x O_D]
        weights = [w[..., lo:hi, :, :] for lo, hi in itertools.pairwise(cache.cum_lengths)]
        weights_dict = {"input_IR": dict(zip(cache.ac_names, weights[:-1])), "observation_IR": weights[-1],}
        for k, v in ensembled_learned_kfs.items(include_nested=True, leaves_only=True):
            ensembled_learned_kfs[k] = utils.rgetitem(weights_dict, k if isinstance(k, str) else ".".join(k)).expand_as(v)

        cache.t += 1
        return error[None], {}, terminate_condition()


class CnnLeastSquaresPretrainPredictor(CnnLeastSquaresPredictor):
    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_least_squares, default_train_func


class CnnAnalyticalPredictor(CnnPredictor):
    @classmethod
    def train_analytical(cls,
                         exclusive: Namespace,
                         ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                         cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: utils.multi_vmap(
                exclusive_.reference_module._analytical_initialization, 2,
                randomness="different",
            )(exclusive_.train_info.systems.td().to_dict()), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_analytical,

    def _analytical_initialization(self, system_state_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        F, H, K = map(system_state_dict["environment"].__getitem__, ("F", "H", "K"))
        B = system_state_dict["environment"].get("B", {})
        S_D = F.shape[0]

        powers = utils.pow_series(F @ (torch.eye(S_D) - K @ H), self.ir_length)                 # [R x S_D x S_D]
        return {
            "input_IR": {
                k: (H @ powers @ B[k]).permute(2, 0, 1)                                         # [I_D x R x O_D]
                for k in vars(self.problem_shape.controller)
            },
            "observation_IR": (H @ powers @ (F @ K)).permute(2, 0, 1)                           # [O_D x R x O_D]
        }, torch.full((), torch.nan)

    def __init__(self, modelArgs: Namespace):
        CnnPredictor.__init__(self, modelArgs)
        self._initialization_error: torch.Tensor = None


class CnnAnalyticalLeastSquaresPredictor(CnnPredictor):
    @classmethod
    def train_analytical_least_squares_newton(cls,
                                              exclusive: Namespace,
                                              ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                                              cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        def newton_analytical(exclusive_: Namespace) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
            # DONE: Remove parameters that do not require gradients before flattening
            _kf_dict = OrderedDict([(k, v) for k, v in ensembled_learned_kfs.items() if v.requires_grad])           # [N x E x F...]
            _flattened_kf_dict = torch.cat([v.flatten(2, -1) for v in _kf_dict.values()], dim=-1)                   # [N x E x F]
            cum_lengths = [0] + np.cumsum([np.prod(v.shape[2:]) for v in _kf_dict.values()]).tolist()

            optimizer = optim.SGD(_kf_dict.values(), lr=0.0)
            def zero_grad_kf_dict() -> None:
                optimizer.zero_grad()
            zero_grad_kf_dict()

            L = ConvolutionalPredictor.analytical_error(ensembled_learned_kfs, exclusive_.train_info.systems.td())["environment", "observation"]    # [N x E]
            L.sum().backward(create_graph=True, retain_graph=True)
            _flattened_kf_grad_dict = torch.cat([v.grad.flatten(2, -1) for v in _kf_dict.values()], dim=-1)         # [N x E x F]
            zero_grad_kf_dict()

            H = []
            for f in range(_flattened_kf_dict.shape[-1]):
                _flattened_kf_grad_dict[:, :, f].sum().backward(create_graph=True, retain_graph=True)               # [N x E]
                H.append(torch.cat([v.grad.flatten(2, -1) for v in _kf_dict.values()], dim=-1))                     # [N x E x F]
                zero_grad_kf_dict()
            H = torch.stack(H, dim=-2)                                                                              # [N x E x F x F]
            assert torch.allclose(H, H.mT), f"Computed Hessian must be symmetric up to numerical precision but got {H}."

            _flattened_newton_step = (torch.inverse(H) @ _flattened_kf_grad_dict.unsqueeze(-1)).squeeze(-1)         # [N x E x F]
            _newton_step = {
                k: _flattened_newton_step[:, :, cum_lengths[i]:cum_lengths[i + 1]].view_as(v)
                for i, (k, v) in enumerate(_kf_dict.items())
            }

            return TensorDict.from_dict({
                (*k.split("."),): v - _newton_step.get(k, 0.)
                for k, v in ensembled_learned_kfs.items(include_nested=True, leaves_only=True)
            }, batch_size=ensembled_learned_kfs.shape), torch.full((), torch.nan)

        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, newton_analytical, cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_analytical_least_squares_newton,


class CnnLeastSquaresRandomStepPredictor(CnnLeastSquaresPredictor):
    @classmethod
    def train_random_step(cls,
                          exclusive: Namespace,
                          ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                          cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: ({
                k: v + torch.normal(0., torch.abs(ensembled_learned_kfs[k].data - v))
                for k, v in cls.vmap_train_least_squares(exclusive_)[0].items()
            }, torch.full((), torch.nan)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_least_squares, default_train_func, cls.train_random_step


class CnnLeastSquaresNegationPredictor(CnnLeastSquaresPredictor):
    @classmethod
    def train_negation(cls,
                       exclusive: Namespace,
                       ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                       cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: ({
                k: 2 * v - ensembled_learned_kfs[k].data
                for k, v in cls.vmap_train_least_squares(exclusive_)[0].items()
            }, torch.full((), torch.nan)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_least_squares, default_train_func, cls.train_negation




