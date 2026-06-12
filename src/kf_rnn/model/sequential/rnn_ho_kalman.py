from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.sequential.rnn_predictor import RnnPredictor
from kf_rnn.model.convolutional.base import ConvolutionalPredictor
from kf_rnn.model.convolutional.cnn_predictor import (
    CnnAnalyticalPredictor,
    CnnAnalyticalLeastSquaresPredictor,
)
from kf_rnn.infrastructure.config.schema import controller_dims


class RnnHoKalmanBasePredictor(RnnPredictor):
    @dataclass
    class Config(RnnPredictor.Config, CnnAnalyticalLeastSquaresPredictor.Config):
        """Carries both the RNN hyperparameters and those of the FIR submodule
        (``ir_length`` / ``ridge``), which is constructed from the same config."""

    FIR_CLS: type[ConvolutionalPredictor] = None    # TODO: Subclasses of Ho-Kalman Predictor need to explicitly state the FIR class
    FIR_ATTR: str = "fir"

    def __init__(self, modelArgs: "RnnHoKalmanBasePredictor.Config", **kwargs: Any):
        if self.FIR_CLS is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set the ``FIR_CLS`` class attribute to a concrete "
                f"ConvolutionalPredictor subclass before instantiation."
            )
        RnnPredictor.__init__(self, modelArgs, **kwargs)
        self.register_module(RnnHoKalmanBasePredictor.FIR_ATTR, self.FIR_CLS(modelArgs, **kwargs))

    def convert_fir(self, stacked_modules: TensorDict, exclusive: SimpleNamespace) -> tuple[dict[str, Any], torch.Tensor]:
        O_D = self.problem_shape.environment.observation
        controller_keys = [*controller_dims(self.problem_shape).keys()]
        ir_length: int = self.get_submodule(RnnHoKalmanBasePredictor.FIR_ATTR).ir_length
        D = max(2 * self.S_D, ir_length)
        d1 = d2 = eu.ceildiv(D, 2)

        fir_weights = stacked_modules[RnnHoKalmanBasePredictor.FIR_ATTR]
        observation_ir: torch.Tensor = fir_weights["observation_IR"]
        input_ir: TensorDict = fir_weights.get("input_IR", TensorDict({}, batch_size=stacked_modules.shape))

        # SUBSECTION: Concatenate impulse responses
        irs = [observation_ir, *map(input_ir.__getitem__, controller_keys)]
        concatenated_irs = torch.cat(irs, dim=-1)
        padded_irs = Fn.pad(concatenated_irs, (0, 0, 0, 1,), mode="constant", value=0.0)
        cum_lengths = np.cumsum([ir.shape[-1] for ir in irs]).tolist()

        # SUBSECTION: Construct the block Hankel matrix
        hankel_indices = (torch.arange(d1)[:, None] + torch.arange(d2 + 1)[None, :]).clamp_max_(ir_length)
        hankel_matrix = padded_irs[..., hankel_indices, :]
        hankel_neg = einops.rearrange(hankel_matrix[..., :, :, :-1, :], "... o d1 d2 i -> ... (d1 o) (d2 i)")   # float: [... x (d1 O_D) x (d2 ?+)]
        hankel_pos = einops.rearrange(hankel_matrix[..., :, :, 1:, :], "... o d1 d2 i -> ... (d1 o) (d2 i)")    # float: [... x (d1 O_D) x (d2 ?+)]

        U, S, Vh = torch.linalg.svd(hankel_neg, full_matrices=False)                    # float: [... x (d1 O_D) x r], [... x r], [... x r x (d2 ?+)]
        V = Vh.mT                                                                       # float: [... x (d2 ?+) x r]
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

    def training_recipe(self) -> Sequence[Any]:
        # Run the FIR submodule's own stages (retargeted onto the ``fir`` submodule),
        # then recover the state-space realization via the Ho-Kalman stage.
        from kf_rnn.infrastructure.experiment.stages import SubmoduleStage, build_stages
        fir = self.get_submodule(RnnHoKalmanBasePredictor.FIR_ATTR)
        fir_stages = build_stages(fir.training_recipe(), fir)
        return [
            SubmoduleStage(stage, RnnHoKalmanBasePredictor.FIR_ATTR)
            for stage in fir_stages
        ] + ["ho_kalman"]


class RnnHoKalmanAnalyticalPredictor(RnnHoKalmanBasePredictor):
    FIR_CLS: type[ConvolutionalPredictor] = CnnAnalyticalPredictor


class RnnHoKalmanAnalyticalLeastSquaresPredictor(RnnHoKalmanAnalyticalPredictor):
    FIR_CLS: type[ConvolutionalPredictor] = CnnAnalyticalLeastSquaresPredictor




