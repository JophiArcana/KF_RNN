"""Equivalence + gradient checks for SequentialPredictor's diagonalized parallel
scan against a naive sequential Kalman loop.

Run: PYTHONPATH=. python tests/test_sequential_scan.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from tensordict import TensorDict

from kf_rnn.infrastructure.config.schema import EnvironmentShape, ProblemShape
from kf_rnn.model.sequential.rnn_predictor import RnnPredictor


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def _build_model(n: int, O_D: int, controller: dict[str, int]) -> RnnPredictor:
    ps = ProblemShape(environment=EnvironmentShape(observation=O_D), controller=controller)
    initialization = {
        "F": 0.5 * torch.randn(n, n),
        "H": torch.randn(O_D, n),
        "K": 0.1 * torch.randn(n, O_D),
        "B": {k: torch.randn(n, d) for k, d in controller.items()},
    }
    model = RnnPredictor(RnnPredictor.Config(problem_shape=ps, S_D=n), **initialization)
    model.eval()
    return model


def _naive(model: RnnPredictor, s0: torch.Tensor, actions: TensorDict, observations: torch.Tensor):
    """Reference Kalman sweep, exactly mirroring SequentialPredictor._forward, but
    written to support arbitrary leading batch dims."""
    L = observations.shape[-2]
    state = s0
    states, obs = [], []
    for t in range(L):
        Bu = sum((actions[k][..., t, :] @ model.B[k].mT for k in model.B), start=torch.zeros_like(s0))
        prior = state @ model.F.mT + Bu
        observation_estimation = prior @ model.H.mT
        state = prior + (observations[..., t, :] - observation_estimation) @ model.K.mT
        states.append(state)
        obs.append(observation_estimation)
    return torch.stack(states, dim=-2), torch.stack(obs, dim=-2)


def _params(model: RnnPredictor):
    return [model.F, model.H, model.K, *model.B.values()]


def _check(name: str, batch: tuple[int, ...], L: int, n: int, O_D: int, controller: dict[str, int]):
    model = _build_model(n, O_D, controller)
    I_D = next(iter(controller.values()), None)

    s0 = torch.randn(*batch, n, requires_grad=True)
    actions = TensorDict(
        {k: torch.randn(*batch, L, d) for k, d in controller.items()},
        batch_size=(*batch, L),
    )
    observations = torch.randn(*batch, L, O_D)

    for p in _params(model):
        p.requires_grad_(True)
    inputs = (*_params(model), s0)

    # SUBSECTION: forward equivalence
    scan = model.forward_with_initial(s0, actions, observations)
    scan_state, scan_obs = scan["environment"]["state"], scan["environment"]["observation"]
    ref_state, ref_obs = _naive(model, s0, actions, observations)

    state_err = (scan_state - ref_state).abs().max().item()
    obs_err = (scan_obs - ref_obs).abs().max().item()
    assert state_err < 1e-9, f"{name}: state fwd err {state_err}"
    assert obs_err < 1e-9, f"{name}: observation fwd err {obs_err}"

    # SUBSECTION: gradient equivalence (random cotangents on both outputs)
    g_state, g_obs = torch.randn_like(scan_state), torch.randn_like(scan_obs)
    scan_loss = (g_state * scan_state).sum() + (g_obs * scan_obs).sum()
    ref_loss = (g_state * ref_state).sum() + (g_obs * ref_obs).sum()

    scan_grads = torch.autograd.grad(scan_loss, inputs, retain_graph=True)
    ref_grads = torch.autograd.grad(ref_loss, inputs)

    grad_err = max((sg - rg).abs().max().item() for sg, rg in zip(scan_grads, ref_grads))
    assert grad_err < 1e-7, f"{name}: grad err {grad_err}"

    print(f"OK  {name:34s} state={state_err:.2e} obs={obs_err:.2e} grad={grad_err:.2e}")


def test_scan_matches_naive_loop() -> None:
    _check("with-control", (5,), 11, 4, 2, {"input": 3})
    _check("no-control", (5,), 11, 4, 2, {})
    _check("multi-batch-dims", (3, 5), 17, 4, 2, {"input": 3})
    _check("length-one", (5,), 1, 4, 2, {"input": 3})
    _check("long-sequence", (4,), 128, 4, 2, {"input": 3})


if __name__ == "__main__":
    test_scan_matches_naive_loop()
    print("ALL_OK")
