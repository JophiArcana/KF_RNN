from argparse import Namespace
from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
# from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
from tensordict import TensorDict, NonTensorData

from infrastructure import utils
from infrastructure.static import ModelPair, TrainFunc
from model.base import Predictor
from model.sequential.base import SequentialPredictor
from model.least_squares_predictor import LeastSquaresPredictor


class RnnPredictor(SequentialPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: torch.Tensor | nn.Parameter):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        self.F = nn.Parameter(initialization.get("F", (1 - self.eps) * torch.eye(self.S_D)))
        self.B = nn.ParameterDict({
            k: nn.Parameter(utils.rgetitem(initialization, f"B.{k}", torch.zeros((self.S_D, d,))))
            for k, d in vars(self.problem_shape.controller).items()
        })
        self.H = nn.Parameter(initialization.get("H", nn.init.kaiming_normal_(torch.zeros((self.O_D, self.S_D,)))))
        self.K = nn.Parameter(initialization.get("K", torch.zeros((self.S_D, self.O_D,))))


class RnnKalmanPredictor(RnnPredictor):
    @classmethod
    def train_analytical(cls,
                         THP: Namespace,
                         exclusive: Namespace,
                         model_pair: ModelPair,
                         cache: Namespace
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return Predictor._train_with_initialization_and_error(
            exclusive, model_pair[1],
            lambda stacked_modules_, exclusive_: ({
                **exclusive_.train_info.systems.td().get("environment", {}),
                **exclusive_.train_info.systems.td().get("controller", {})
            }, Predictor.evaluate_run(
                exclusive_.train_info.dataset.obj["environment", "target_observation_estimation"],
                exclusive_.train_info.dataset.obj, ("environment", "observation")
            ).squeeze(-1)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (cls.train_analytical, Predictor.terminate_with_initialization_and_error,),


class RnnKalmanInitializedPredictor(RnnKalmanPredictor):
    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnKalmanPredictor.train_func_list(default_train_func) + (default_train_func,)


class RnnComplexDiagonalPredictor(SequentialPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: torch.Tensor | nn.Parameter):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        keys = ("F", "B", "H", "K",)
        initalized_keys = (*filter(initialization.__contains__, keys),)
        assert len(initalized_keys) in (0, len(keys),)

        if len(initalized_keys) == 0:
            P_init = torch.zeros((self.S_D, self.O_D,))
            B_init = {
                k: torch.zeros((self.S_D, d,))
                for k, d in vars(self.problem_shape.controller).items()
            }
            H_init = nn.init.kaiming_normal_(torch.zeros((self.O_D, self.S_D,)))
            logD_init = torch.complex(
                torch.full((self.S_D,), -self.eps),
                torch.zeros((self.S_D,)).uniform_(-torch.pi / 10, torch.pi / 10),
            )
        else:
            F, B, H, K = map(initialization.__getitem__, keys)
            FK = F @ K
            M = F - FK @ H
            L, V = torch.linalg.eig(M)
            Vinv = torch.inverse(V)

            P_init = Vinv @ FK
            B_init = {k: Vinv @ _B for k, _B in B.items()}
            H_init = H @ V
            logD_init = torch.log(utils.complex(L))
        
        self.P = nn.Parameter(utils.complex(P_init))
        self.B = nn.ParameterDict({k: nn.Parameter(utils.complex(v)) for k, v in B_init.items()})
        self.H = nn.Parameter(utils.complex(H_init))
        self.logD = nn.Parameter(logD_init)

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> tuple[TensorDict, Namespace]:                       # [B...]
        kfs = kfs.clone()
        F = torch.diag_embed(torch.exp(kfs["logD"])) + kfs["P"] @ kfs["H"]
        kfs["F"] = F
        kfs["K"] = torch.inverse(F) @ kfs["P"]
        return SequentialPredictor._analytical_error_and_cache(kfs, systems)

    def forward(self, trace: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        trace: TensorDict = self.trace_to_td(trace)
        actions, observations = trace["controller"], trace["environment", "observation"]    # [B... x L x I_D?], [B... x L x O_D]

        L = trace.shape[-1]
        state_initialization = self.sample_initial_as_observations(observations, (*trace.shape[:-1], self.S_D,))    # [B... x S_D]

        observation_embds = utils.complex(torch.cat((
            torch.zeros_like(observations[..., :1, :]),
            observations[..., :-1, :],
        ), dim=-2)) @ self.P.mT                                                                                     # [B... x L x S_D]
        action_embds = sum(utils.complex(ac) @ self.B[ac_name].mT for ac_name, ac in actions.items())               # [B... x L x S_D]
        embds = observation_embds + action_embds                                                                    # [B... x L x S_D]
        embds = torch.cat((state_initialization[..., None, :], embds,), dim=-2)                                     # [B... x (L + 1) x S_D]

        # weights = torch.exp(self.logD * torch.arange(L, -1, -1)[:, None])
        # state_estimations = (torch.cumsum(embds * weights, dim=-2) / weights)[..., 1:, :]
        k = 1
        while k < (L + 1):
            w = torch.exp(self.logD * k)
            embds = torch.cat((embds[..., :k, :], embds[..., k:, :] + w * embds[..., :-k, :]), dim=-2)
            k <<= 1
        state_estimations = embds[..., 1:, :]
        observation_estimations = state_estimations @ self.H.mT

        if not self.training:
            state_estimations = torch.real(state_estimations)
            observation_estimations = torch.real(observation_estimations)

        return {
            "environment": {
                "state": state_estimations,
                "observation": observation_estimations,
            },
            "controller": {},
        }



# class RnnLeastSquaresPredictor(RnnPredictor, LeastSquaresPredictor):
#     def __init__(self, modelArgs: Namespace):
#         RnnPredictor.__init__(self, modelArgs)
#         LeastSquaresPredictor.__init__(self, modelArgs)
#
#     def _least_squares_initialization(self, trace: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
#         trace = self.trace_to_td(trace).flatten(0, -2)
#         actions, observations = trace["controller"], trace["environment"]["observation"]
#
#         bsz, L = observations.shape[:2]
#         ac_names = list(actions.keys())
#
#         parameters = utils.parameter_td(self).apply(torch.Tensor.detach)
#         parameter_eqs = parameters.apply(lambda t: NonTensorData(None), batch_size=())
#
#         def update_parameter_eqs(k_: Union[str, tuple[str, ...]], X_: torch.Tensor, y_: torch.Tensor) -> None:
#             eqs = parameter_eqs[k_]
#             y_ = y_[..., None]
#             XTX_, XTy_ = X_.mT @ X_, X_.mT @ y_
#             if eqs is None:
#                 parameter_eqs[k_] = {"XTX": XTX_ + self.ridge * torch.eye(XTX_.shape[-1]), "XTy": XTy_, "X": X_, "y": y_}
#             else:
#                 eqs["XTX"] = eqs["XTX"] + XTX_
#                 eqs["XTy"] = eqs["XTy"] + XTy_
#                 if eqs["X"] is not None and eqs["y"] is not None:
#                     eqs["X"] = torch.cat((eqs["X"], X_), dim=-2)
#                     eqs["y"] = torch.cat((eqs["y"], y_), dim=-2)
#
#         state = self.sample_initial_as_observations(None, (bsz, self.S_D))      # [bsz x S_D]
#         # state = torch.zeros((bsz, self.S_D))                                    # [bsz x O_D]
#         error = torch.zeros((bsz, self.O_D))                                    # [bsz x O_D]
#
#         required_eqs = {
#             "F": utils.ceildiv(self.S_D ** 2, bsz * self.O_D) + 1,
#             "H": 1 if bsz >= self.S_D else (utils.ceildiv(self.S_D, bsz) + 1),
#             "K": utils.ceildiv(self.S_D, bsz) + 1, **{
#                 ("B", ac_name): utils.ceildiv(self.S_D * getattr(self.problem_shape.controller, ac_name), bsz * self.O_D) + 1
#                 for ac_name in ac_names
#             }
#         }
#
#         for it, (action, observation) in enumerate(zip(                         # [bsz x ?], [bsz x O_D]
#             TensorDict.unbind(actions, dim=1),
#             torch.unbind(observations, dim=1),
#         )):
#             # print(f"Iteration {it}:", dict([(k, v.norm()) for k, v in parameters.items()]))
#             # if it == 10:
#             #     raise Exception()
#             F, H, K = map(parameters.__getitem__, ("F", "H", "K"))
#             B = parameters.get("B", TensorDict({}, batch_size=()))
#
#             updated_state = state + error @ K.mT                                # [bsz x S_D]
#             Fx = updated_state @ F.mT                                           # [bsz x S_D]
#             Bu = torch.zeros((bsz, self.S_D)) + sum(
#                 action[ac_name] @ B[ac_name].mT
#                 for ac_name in ac_names
#             )                                                                   # [bsz x S_D]
#             next_state = Fx + Bu                                                # [bsz x S_D]
#
#             # SECTION: Compute least squares equations for each parameter
#             update_parameter_eqs("H", next_state[None], observation.mT)         # [1 x bsz x S_D], [O_D x bsz]
#             if it > 0:
#                 update_parameter_eqs("F", einops.rearrange(                         # [(O_D * bsz) x (S_D^2)], [O_D * bsz]
#                     H[:, None, :, None] * updated_state[None, :, None, :],
#                     "o bsz s1 s2 -> (o bsz) (s1 s2)"
#                 ), (observation - Bu @ H.mT).mT.flatten())
#                 update_parameter_eqs("K", einops.rearrange(                         # [(O_D * bsz) x (S_D * O_D)], [O_D * bsz]
#                     (H @ F)[:, None, :, None] * error[None, :, None, :],
#                     "o1 bsz s o2 -> (o1 bsz) (s o2)"
#                 ), (observation - (state + Bu) @ H.mT).mT.flatten())
#                 for ac_name in ac_names:
#                     update_parameter_eqs(("B", ac_name), einops.rearrange(          # [(O_D * bsz) x (S_D * I_D)], [O_D * bsz]
#                         H[:, None, :, None] * action[ac_name][None, :, None, :],
#                         "o bsz s i -> (o bsz) (s i)"
#                     ), (observation - Fx @ H.mT).mT.flatten())
#
#             # SECTION: Update running variables
#             error = observation - state @ H.mT
#             state = next_state
#
#             # rank_dict = {}
#             # for k, eqs in parameter_eqs.items():
#             #     try:
#             #         rank_dict[k] = (torch.linalg.matrix_rank(eqs["XTX"]), eqs["XTX"].shape[-1])
#             #     except Exception:
#             #         pass
#
#             for k, eqs in parameter_eqs.items():
#                 if it + 1 >= required_eqs[k]:
#                     XTX, XTy = eqs["XTX"], eqs["XTy"]                           # [d x d], [d]
#                     next_parameter = torch.inverse(XTX) @ XTy
#                 elif it > 0 or k == "H":
#                     X, y = eqs["X"], eqs["y"]                                   # [n x d], [n]
#                     next_parameter = X.mT @ torch.inverse(X @ X.mT) @ y
#                 else:
#                     next_parameter = parameters[k]
#
#                 # if k == "F" and it == 1:
#                 #     print(eqs["X"], eqs["y"])
#                 #     print(torch.linalg.matrix_rank(eqs["X"] @ eqs["X"].mT))
#                 #     print(torch.linalg.matrix_rank(eqs["XTX"]))
#                 #     # print(torch.inverse(eqs["XTX"]))
#                 #     raise Exception()
#                 # try:
#                 #     XTX, XTy = eqs["XTX"], eqs["XTy"]                           # [d x d], [d]
#                 #     next_parameter = torch.inverse(XTX) @ XTy
#                 #     eqs["X"] = eqs["y"] = None
#                 #     print(k, "option 1")
#                 # except Exception:
#                 #     try:
#                 #         X, y = eqs["X"], eqs["y"]                               # [n x d], [n]
#                 #         next_parameter = X.mT @ torch.inverse(X @ X.mT + self.ridge * torch.eye(X.shape[-2])) @ y
#                 #         print(k, "option 2")
#                 #     except Exception:
#                 #         next_parameter = parameters[k]
#                 #         print(k, "option 3")
#                 parameters[k] = next_parameter.reshape_as(parameters[k])
#
#         return parameters.to_dict(), torch.full((), torch.nan)






