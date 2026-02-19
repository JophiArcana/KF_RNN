from argparse import Namespace
from typing import Any, Callable, Sequence, TypeVar

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.static import ModelPair, TrainFunc
from model.base import Predictor
from model.sequential.rnn_predictor import RnnPredictor
from model.convolutional.base import ConvolutionalPredictor
from model.convolutional.cnn_predictor import (
    CnnAnalyticalPredictor,
    CnnAnalyticalLeastSquaresPredictor,
)


class RnnHoKalmanBasePredictor(RnnPredictor):
    FIR_CLS: type[ConvolutionalPredictor] = None    # TODO: Subclasses of Ho-Kalman Predictor need to explicitly state the FIR class
    FIR_ATTR: str = "fir"

    def __init__(self, modelArgs: Namespace, **kwargs: Any):
        RnnPredictor.__init__(self, modelArgs, **kwargs)
        self.register_module(RnnHoKalmanBasePredictor.FIR_ATTR, self.FIR_CLS(modelArgs, **kwargs))

    @classmethod
    def terminate_ho_kalman(
            cls,
            THP: Namespace,
            exclusive: Namespace,
            model_pair: ModelPair,
            cache: Namespace,
    ) -> bool:
        return getattr(cache, "done", False)

    @classmethod
    def train_ho_kalman(cls,
                        THP: Namespace,
                        exclusive: Namespace,
                        model_pair: ModelPair,
                        cache: Namespace,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference_module, stacked_modules = model_pair
        return Predictor._train_with_initialization_and_error(
            exclusive, stacked_modules, reference_module.convert_fir, cache
        )

    def convert_fir(self, stacked_modules: TensorDict, exclusive: Namespace) -> tuple[dict[str, Any], torch.Tensor]:
        O_D = self.problem_shape.environment.observation
        controller_keys = [*vars(self.problem_shape.controller).keys()]
        ir_length: int = self.get_submodule(RnnHoKalmanBasePredictor.FIR_ATTR).ir_length
        D = max(2 * self.S_D, ir_length)
        d1 = d2 = utils.ceildiv(D, 2)

        fir_weights = stacked_modules[RnnHoKalmanBasePredictor.FIR_ATTR]
        observation_ir: torch.Tensor = fir_weights["observation_IR"]
        input_ir: TensorDict = fir_weights.get("input_IR", TensorDict({}, batch_size=stacked_modules.shape))

        # SUBSECTION: Concatenate impulse responses
        irs = [observation_ir, *map(input_ir.__getitem__, controller_keys)]
        concatenated_irs = torch.cat([observation_ir, *map(input_ir.__getitem__, controller_keys)], dim=-1)
        padded_irs = Fn.pad(concatenated_irs, (0, 0, 0, 1,), mode="constant", value=0.0)
        cum_lengths = np.cumsum([ir.shape[-1] for ir in irs]).tolist()

        # SUBSECTION: Construct the block Hankel matrix
        hankel_indices = (torch.arange(d1)[:, None] + torch.arange(d2 + 1)[None, :]).clamp_max_(ir_length)
        hankel_matrix = padded_irs[..., hankel_indices, :]
        hankel_neg = einops.rearrange(hankel_matrix[..., :, :, :-1, :], "... o d1 d2 i -> ... (d1 o) (d2 i)")   # float: [... x (d1 O_D) x (d2 ?+)]
        hankel_pos = einops.rearrange(hankel_matrix[..., :, :, 1:, :], "... o d1 d2 i -> ... (d1 o) (d2 i)")    # float: [... x (d1 O_D) x (d2 ?+)]

        U, S, V = torch.svd(hankel_neg, some=True)                                      # float: [... x (d1 O_D) x (d1 O_D)], [... x (d1 O_D)], [... x (d2 ?+) x (d1 O_D)]
        U, S, V = U[..., :self.S_D], S[..., :self.S_D], V[..., :self.S_D]               # float: [... x (d1 O_D) x S_D], [... x S_D], [... x (d2 ?+) x S_D]
        sqrt_S = torch.sqrt(S)                                                          # float: [... x S_D]

        Ch = U[..., :concatenated_irs.shape[-3], :] * sqrt_S[..., None, :]              # float: [... x O_D x S_D]
        Bh = V[..., :concatenated_irs.shape[-1], :] * sqrt_S[..., None, :]              # float: [... x ?+ x S_D]
        Ah = ((U.mT @ hankel_pos @ V) / (sqrt_S[..., :, None] * sqrt_S[..., None, :])).nan_to_num(nan=0.0)  # float: [... x S_D x S_D]

        # SUBSECTION: Extract F, H, K, B from the Ho-Kalman output
        H = Ch                                              # float: [... x O_D x S_D]
        FK = Bh[..., :O_D, :].mT                            # float: [... x S_D x O_D]
        B = {
            ac_name: Bh[..., cum_lengths[i]:cum_lengths[i + 1], :].mT
            for i, ac_name in enumerate(controller_keys)    # float: [... x S_D x I_D]
        }
        F = Ah + FK @ H                                     # float: [... x S_D x S_D]
        K = torch.linalg.pinv(F) @ FK                       # float: [... x S_D x O_D]

        return {"F": F, "H": H, "K": K, "B": B,}, torch.full((), torch.nan)

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        ReturnType = TypeVar("ReturnType")

        def augment_fn(fn: Callable[[Namespace, Namespace, ModelPair, Namespace], ReturnType]) -> Callable[[Namespace, Namespace, ModelPair, Namespace], ReturnType]:
            def augmented_fn(THP: Namespace, exclusive: Namespace, model_pair: ModelPair, cache: Namespace) -> ReturnType:
                return fn(THP, exclusive, (
                    model_pair[0].get_submodule(RnnHoKalmanBasePredictor.FIR_ATTR),
                    model_pair[1][RnnHoKalmanBasePredictor.FIR_ATTR],
                ), cache)
            return augmented_fn

        train_func_list = []
        for training_func, terminate_func in cls.FIR_CLS.train_func_list(default_train_func):
            train_func_list.append((augment_fn(training_func), augment_fn(terminate_func),))
        train_func_list.append((RnnHoKalmanBasePredictor.train_ho_kalman, RnnHoKalmanBasePredictor.terminate_ho_kalman,))
        return (*train_func_list,)


class RnnHoKalmanAnalyticalPredictor(RnnHoKalmanBasePredictor):
    FIR_CLS: type[ConvolutionalPredictor] = CnnAnalyticalPredictor


class RnnHoKalmanAnalyticalLeastSquaresPredictor(RnnHoKalmanAnalyticalPredictor):
    FIR_CLS: type[ConvolutionalPredictor] = CnnAnalyticalLeastSquaresPredictor




