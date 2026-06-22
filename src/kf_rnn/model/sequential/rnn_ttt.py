from dataclasses import dataclass
from typing import Any, Sequence

import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.sequential.rnn_predictor import RnnPredictor


class RnnInContextPredictor(RnnPredictor):
    """A Kalman-style predictor used purely in-context (empty training recipe).

    At every timestep it makes the standard one-step prediction with the current
    fast-weights, then takes ``n_steps`` SGD updates (learning rate ``step_size``)
    on a ``0.5 * MSE`` loss over a sliding window of the last ``window``
    observations. Each trajectory in the batch adapts its *own* fast-weights;
    propagation reuses the inherited :class:`RnnPredictor` step verbatim.

    The mechanics are composed from ``ecliseutils`` primitives:

    - propagation is the inherited ``SequentialPredictor._forward`` driven through
      :class:`eu.FunctionalMethod` + ``torch.func.functional_call``, so the base
      RNN step stays oblivious to the fact that its weights are being updated;
    - the online recurrence is an :func:`eu.scan` whose carry is
      ``(theta, filtered_state, entering_state_window)``;
    - per-trajectory adaptation is :func:`eu.multi_vmap` over the data-batch dims.

    Gradient computation (:meth:`_compute_grads`) and the parameter update
    (:meth:`_optimizer_step`) are independent, overridable hooks, so custom
    backprop schemes (straight-through estimators / quantized weights, updating a
    subset of ``theta``, projected updates, ...) drop in without touching the
    propagation or the scan.
    """

    @dataclass
    class Config(RnnPredictor.Config):
        n_steps: int = 1
        step_size: float = 1.0
        window: int = 1

    def __init__(self, modelArgs: "RnnInContextPredictor.Config", **kwargs: Any):
        RnnPredictor.__init__(self, modelArgs, **kwargs)
        self.n_steps = modelArgs.n_steps
        self.step_size = modelArgs.step_size
        self.window = modelArgs.window
        # Propagation vehicle: exposes the inherited single-step ``_forward`` as a
        # ``functional_call`` target while sharing this module's own parameters, so
        # the fast-weights substituted per step reach ``_forward`` directly (no
        # duplicate matrices). Stored off the module registry to avoid being a
        # submodule of itself.
        object.__setattr__(self, "cell", eu.FunctionalMethod(self, "_forward"))

    def _ttt_loss(self,
                  prediction: TensorDict,
                  action: dict[str, torch.Tensor],
                  observation: torch.Tensor,    # [O_D]
    ) -> torch.Tensor:                          # scalar
        """Per-step self-supervised objective driving the test-time update.

        Defaults to ``0.5 * ||y_hat - y||^2``. Override in subclasses for other
        objectives (e.g. mean negative log-likelihood once an explicit generative
        model with ``S_W`` / ``S_V`` exists, or a reward term involving the
        controller actions)."""
        residual = prediction["environment", "observation"] - observation
        return 0.5 * residual.pow(2).sum(-1)

    def _window_loss(self,
                     theta: dict[str, torch.Tensor],
                     s_start: torch.Tensor,                  # [S_D]
                     win_actions: dict[str, torch.Tensor],   # [w x I_D]
                     win_obs: torch.Tensor,                  # [w x O_D]
    ) -> torch.Tensor:                                       # scalar
        """Window-averaged ``_ttt_loss``: roll ``_forward`` from the (detached)
        window-start state through the stored steps under ``theta``."""
        s = s_start
        total = win_obs.new_zeros(())
        w = win_obs.shape[0]
        for j in range(w):
            action_j = {k: v[j] for k, v in win_actions.items()}
            obs_j = win_obs[j]
            out = torch.func.functional_call(self.cell, theta, (s, action_j, obs_j))
            total = total + self._ttt_loss(out, action_j, obs_j)
            s = out["environment", "state"]
        return total / w

    def _compute_grads(self,
                       theta: dict[str, torch.Tensor],
                       s_start: torch.Tensor,
                       win_actions: dict[str, torch.Tensor],
                       win_obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update direction w.r.t. ``theta``. Override for custom backprop."""
        return torch.func.grad(self._window_loss)(theta, s_start, win_actions, win_obs)

    def _optimizer_step(self,
                        theta: dict[str, torch.Tensor],
                        grads: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply one update. Override for momentum / projected / quantized steps."""
        return eu.sgd_step(theta, grads, self.step_size)

    def forward(self, trace: dict[str, dict[str, torch.Tensor]], mode: str = None) -> dict[str, dict[str, torch.Tensor]]:
        trace: TensorDict = self.trace_to_td(trace)
        actions_td, observations = trace["controller"], trace["environment", "observation"]    # [B... x L x I_D?], [B... x L x O_D]

        # Plain dict of tensors (avoids TensorDict-under-vmap edge cases); keyed by
        # controller name to match the cell's ``B.<name>`` parameters.
        actions_d: dict[str, torch.Tensor] = actions_td.to_dict()

        batch_shape = observations.shape[:-2]
        state_0 = self.sample_initial_as_observations(observations, (*batch_shape, self.S_D,))  # [B... x S_D]

        # Shared fast-weight initialization (the canonical, ensemble-substituted
        # parameters). Inside the vmap these diverge per trajectory.
        theta_init: dict[str, torch.Tensor] = eu.td_items(eu.parameter_td(self).detach())

        def run_sequence(state_0: torch.Tensor,                 # [S_D]
                         actions_d: dict[str, torch.Tensor],    # [L x I_D]
                         observations: torch.Tensor,            # [L x O_D]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            L = observations.shape[0]
            # Rolling window of the last ``window`` *entering* states (left-padded
            # with the initial state), so the window-start state is always the
            # oldest entry after pushing the current one.
            entering_window = state_0[None].expand(self.window, *state_0.shape)     # [w x S_D]

            def step(carry: tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
                     x: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor],
            ):
                theta, s, ent_win = carry
                t, action_t, obs_t = x
                t = int(t)

                # SUBSECTION: standard prediction with the current fast-weights.
                out = torch.func.functional_call(self.cell, theta, (s, action_t, obs_t))
                y_hat = out["environment", "observation"]
                s_post = out["environment", "state"]

                # SUBSECTION: test-time update over the sliding window [w0 .. t].
                ent_win = torch.cat([ent_win[1:], s[None]], dim=0)      # push entering state s
                s_start = ent_win[0].detach()
                w0 = max(0, t - self.window + 1)
                win_actions = {k: v[w0:t + 1] for k, v in actions_d.items()}
                win_obs = observations[w0:t + 1]
                for _ in range(self.n_steps):
                    grads = self._compute_grads(theta, s_start, win_actions, win_obs)
                    theta = self._optimizer_step(theta, grads)

                return (theta, s_post, ent_win), (y_hat, s_post)

            xs = (torch.arange(L), actions_d, observations)
            _, (preds, states) = eu.scan(step, (theta_init, state_0, entering_window), xs)
            return preds, states     # [L x O_D], [L x S_D]

        # Map the per-sequence routine over every leading batch dim so the
        # propagation/update math always sees 2-D weights and a single trajectory.
        mapped = eu.multi_vmap(run_sequence, len(batch_shape), in_dims=0, randomness="different")
        observation_estimations, state_estimations = mapped(state_0, actions_d, observations)

        return {
            "environment": {
                "state": state_estimations,
                "observation": observation_estimations,
            },
            "controller": {},
        }

    def training_recipe(self) -> Sequence[str]:
        return []
