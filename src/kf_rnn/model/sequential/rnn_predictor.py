from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.base import Predictor
from kf_rnn.model.sequential.base import SequentialPredictor
from kf_rnn.infrastructure.config.schema import controller_dims


class RnnPredictor(SequentialPredictor):
    def __init__(self, modelArgs: "RnnPredictor.Config", **initialization: torch.Tensor | nn.Parameter):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        self.F = nn.Parameter(initialization.get("F", (1 - self.eps) * torch.eye(self.S_D)))
        self.B = nn.ParameterDict({
            k: nn.Parameter(eu.rgetitem(initialization, f"B.{k}", torch.zeros((self.S_D, d,))))
            for k, d in controller_dims(self.problem_shape).items()
        })
        self.H = nn.Parameter(initialization.get("H", nn.init.kaiming_normal_(torch.zeros((self.O_D, self.S_D,)))))
        self.K = nn.Parameter(initialization.get("K", torch.zeros((self.S_D, self.O_D,))))


class RnnKalmanPredictor(RnnPredictor):
    def analytical_initialization(self, exclusive: SimpleNamespace) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        systems_td = exclusive.train_info.systems.td()
        initialization = {
            **systems_td.get("environment", {}),
            **systems_td.get("controller", {}),
        }
        error = Predictor.evaluate_run(
            exclusive.train_info.dataset["environment", "target_observation_estimation"],
            exclusive.train_info.dataset, ("environment", "observation")
        ).squeeze(-1)
        return initialization, error

    def training_recipe(self) -> Sequence[str]:
        return ["analytical_init"]


class RnnKalmanInitializedPredictor(RnnKalmanPredictor):
    def training_recipe(self) -> Sequence[str]:
        return ["analytical_init", "sgd"]


class RnnComplexDiagonalPredictor(SequentialPredictor):
    def __init__(self, modelArgs: "RnnComplexDiagonalPredictor.Config", **initialization: torch.Tensor | nn.Parameter):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        keys = ("F", "B", "H", "K",)
        initialized_keys = (*filter(initialization.__contains__, keys),)
        assert len(initialized_keys) in (0, len(keys),)

        if len(initialized_keys) == 0:
            P_init = torch.zeros((self.S_D, self.O_D,))
            B_init = {
                k: torch.zeros((self.S_D, d,))
                for k, d in controller_dims(self.problem_shape).items()
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
            logD_init = torch.log(eu.complex(L))
        
        self.P = nn.Parameter(eu.complex(P_init))
        self.B = nn.ParameterDict({k: nn.Parameter(eu.complex(v)) for k, v in B_init.items()})
        self.H = nn.Parameter(eu.complex(H_init))
        self.logD = nn.Parameter(logD_init)

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> tuple[TensorDict, SimpleNamespace]:                       # [B...]
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

        observation_embds = eu.complex(torch.cat((
            torch.zeros_like(observations[..., :1, :]),
            observations[..., :-1, :],
        ), dim=-2)) @ self.P.mT                                                                                     # [B... x L x S_D]
        action_embds = sum(eu.complex(ac) @ self.B[ac_name].mT for ac_name, ac in actions.items())               # [B... x L x S_D]
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



