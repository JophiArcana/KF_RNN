from argparse import Namespace
from collections import OrderedDict
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.training import TrainFunc
from model.base import Predictor
from model.convolutional.base import ConvolutionalPredictor


class CnnPredictor(ConvolutionalPredictor):
    def __init__(self, modelArgs: Namespace):
        ConvolutionalPredictor.__init__(self, modelArgs)
        self.ir_length = modelArgs.ir_length

        self.input_IR = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((v, self.ir_length, self.O_D)))                         # [? x R x O_D]
            for k, v in vars(self.problem_shape.controller).items()
        })
        self.observation_IR = nn.Parameter(torch.zeros((self.O_D, self.ir_length, self.O_D)))   # [O_D x R x O_D]


class CnnPredictorLeastSquares(CnnPredictor):
    @classmethod
    def train_least_squares(cls,
                            exclusive: Namespace,
                            ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                            cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs,
            CnnPredictorLeastSquares.vmap_train_least_squares, cache
        )

    @classmethod
    def vmap_train_least_squares(cls, exclusive_: Namespace) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return utils.double_vmap(exclusive_.reference_module._least_squares_initialization)(exclusive_.train_info.dataset.obj.to_dict())

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnPredictorLeastSquares.train_least_squares,

    """ forward
        :parameter {
            "input": [B x L x I_D],
            "observation": [B x L x O_D]
        }
        :returns None
    """
    def _least_squares_initialization(self, trace: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        trace = self.trace_to_td(trace).flatten(0, -2)
        actions, observations = trace["controller"], trace["environment"]["observation"]

        B, L = observations.shape[:2]

        # DONE: Implement online least squares for memory efficiency
        split = [0]
        while split[-1] != L:
            split.append(min(split[-1] + min(L, 1 << 16), L))

        padded_observations = torch.cat([                                                       # [B x L x O_D]
            torch.zeros((B, 1, self.O_D)),
            observations[:, :-1],
            torch.zeros((B, 1, self.O_D))
        ], dim=1)
        padded_actions = torch.cat([
            actions, actions[:, -1:].apply(torch.zeros_like)
        ], dim=1)                                                                               # [B x L x ...]

        ac_names = list(padded_actions.keys())
        flattened_padded_data = torch.cat([
            *map(padded_actions.__getitem__, ac_names), padded_observations
        ], dim=-1)                                                                              # [B x L x F]
        cum_lengths = [0] + np.cumsum([*map(vars(self.problem_shape.controller).__getitem__, ac_names), self.O_D]).tolist()

        XTX = torch.zeros((r_ := self.ir_length * cum_lengths[-1], r_))                         # [RF x RF]
        XTy = torch.zeros((r_, self.O_D))                                                       # [RF x O_D]
        yTy = torch.zeros((self.O_D, self.O_D))                                                 # [O_D x O_D]

        for i in range(len(split) - 1):
            lo, hi = split[i], split[i + 1]
            l = hi - lo

            indices = (torch.arange(lo, hi)[:, None] - torch.arange(self.ir_length)).clamp_min(-1)  # [l x R]

            X = flattened_padded_data[:, indices]                                               # [B x l x R x F]
            flattened_X = X.view((B * l, self.ir_length * cum_lengths[-1]))                     # [Bl x RF]
            flattened_observations = observations[:, lo:hi].reshape((B * l, self.O_D))          # [Bl x O_D]

            XTX = XTX + (flattened_X.mT @ flattened_X)
            XTy = XTy + (flattened_X.mT @ flattened_observations)
            yTy = yTy + (flattened_observations.mT @ flattened_observations)

            torch.cuda.empty_cache()

        XTX_lI_inv = torch.inverse(XTX + self.ridge * torch.eye(r_))                            # [RF x RF]
        flattened_w = XTX_lI_inv @ XTy
        w = flattened_w.unflatten(0, (self.ir_length, -1)).transpose(0, 1)                      # [F x R x O_D]


        error = torch.trace(yTy + XTy.mT @ (XTX_lI_inv @ XTX @ XTX_lI_inv - 2 * XTX_lI_inv) @ XTy) / (B * L)
        return {
            "input_IR": {
                ac_name: w[cum_lengths[idx]:cum_lengths[idx + 1]]
                for idx, ac_name in enumerate(ac_names)
            },
            "observation_IR": w[-self.O_D:]
        }, error

    def __init__(self, modelArgs: Namespace):
        CnnPredictor.__init__(self, modelArgs)
        self.ridge = getattr(modelArgs, "ridge", 0.)


class CnnPredictorPretrainLeastSquares(CnnPredictorLeastSquares):
    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnPredictorLeastSquares.train_least_squares, default_train_func


class CnnPredictorAnalytical(CnnPredictor):
    @classmethod
    def train_analytical(cls,
                         exclusive: Namespace,
                         ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                         cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: utils.double_vmap(exclusive_.reference_module._analytical_initialization)(
                exclusive_.train_info.systems.td().to_dict()
            ), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnPredictorAnalytical.train_analytical,

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


class CnnPredictorAnalyticalLeastSquares(CnnPredictor):
    @classmethod
    def train_analytical_least_squares_newton(cls,
                                              exclusive: Namespace,
                                              ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                                              cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
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
        return CnnPredictorAnalyticalLeastSquares.train_analytical_least_squares_newton,


class CnnPredictorLeastSquaresRandomStep(CnnPredictorLeastSquares):
    @classmethod
    def train_random_step(cls,
                          exclusive: Namespace,
                          ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                          cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: ({
                k: v + torch.normal(0., torch.abs(ensembled_learned_kfs[k].data - v))
                for k, v in CnnPredictorLeastSquares.vmap_train_least_squares(exclusive_)[0].items()
            }, torch.full((), torch.nan)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnPredictorLeastSquares.train_least_squares, default_train_func, CnnPredictorLeastSquaresRandomStep.train_random_step


class CnnPredictorLeastSquaresNegation(CnnPredictorLeastSquares):
    @classmethod
    def train_negation(cls,
                       exclusive: Namespace,
                       ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                       cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs, lambda exclusive_: ({
                k: 2 * v - ensembled_learned_kfs[k].data
                for k, v in CnnPredictorLeastSquares.vmap_train_least_squares(exclusive_)[0].items()
            }, torch.full((), torch.nan)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnPredictorLeastSquares.train_least_squares, default_train_func, CnnPredictorLeastSquaresNegation.train_negation




