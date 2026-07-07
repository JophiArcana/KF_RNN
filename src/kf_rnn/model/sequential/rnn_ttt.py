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
        step_decay: float = 0.0          # exponent ``a`` in ``eta_t = step_size * (t+1)^(-a)``; 0.0 => constant step
        polyak_burnin: float = -1.0      # tail-average start as a fraction of L; < 0 => report the raw iterate

    def __init__(self, modelArgs: "RnnInContextPredictor.Config", **kwargs: Any):
        RnnPredictor.__init__(self, modelArgs, **kwargs)
        self.n_steps = modelArgs.n_steps
        self.step_size = modelArgs.step_size
        self.window = modelArgs.window
        self.step_decay = modelArgs.step_decay
        self.polyak_burnin = modelArgs.polyak_burnin
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

    def _step_size_at(self, t: int) -> float:
        """Online learning rate at global step ``t``: ``step_size * (t+1)^(-step_decay)``.

        With the default ``step_decay = 0`` this is the constant ``step_size``; a
        positive exponent ``a in (1/2, 1)`` makes the gain vanish (Robbins-Monro),
        the prerequisite for sub-constant regret rather than a constant noise ball."""
        return self.step_size * (t + 1) ** (-self.step_decay)

    def _optimizer_step(self,
                        theta: dict[str, torch.Tensor],
                        grads: dict[str, torch.Tensor],
                        t: int,
    ) -> dict[str, torch.Tensor]:
        """Apply one update. Override for momentum / projected / quantized steps."""
        return eu.sgd_step(theta, grads, self._step_size_at(t))

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
                    theta = self._optimizer_step(theta, grads, t)

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


class RnnSelfDistillPredictor(RnnInContextPredictor):
    """In-context (TTT) filter trained by the three section-6 prediction losses.

    Reuses :class:`RnnInContextPredictor`'s online recurrence verbatim (the
    Kalman step in :meth:`SequentialPredictor._forward`); only the per-window
    objective and the parameter update are overridden. Every per-step quantity
    needed already exists in the step output ``out`` (no second recurrence):

    - prior state    ``s_prior = F x_{t-1}^+``        (recovered below)
    - a-priori obs    ``y_hat  = H F x_{t-1}^+``        (``out["..","observation"]``)
    - posterior state ``s_post = x_t^+``                (``out["..","state"]``)
    - a-posteriori obs ``y_post = H x_t^+``             (decoded with ``theta["H"]``)

    The window-averaged loss is the weighted sum (DEFAULT weights reproduce the
    a-priori-only M4 objective of the parent, so behaviour is unchanged unless
    the new weights are set)::

        L =  beta0 * 0.5||y_hat  - y||^2    # a-priori observation prediction (grounding; M4)
          +  beta2 * 0.5||y_post - y||^2    # a-posteriori observation prediction (direct-K; section 6.1)
          +  alpha * sum_{k=1}^{n} 0.5||sg(x_t^+) - F^k x_{t-k}^+||^2   # n-step latent self-distillation (section 4.2/6; stop-grad target)

    The stop-gradient on the latent target mirrors
    ``scripts/self_distillation_eigenphase.py``: the gradient flows only through
    the autonomous a-priori roll-out ``F^k x_{t-k}^+``, nudging ``F`` to absorb the
    correction (TD(0) bootstrap of the latent dynamics). ``sd_horizon = n`` sets
    the ladder depth (``n = 1`` is the original single-step ``F x_{t-1}^+`` term);
    ``keep_launch`` toggles whether the (non-root) launch state carries gradient.

    Combined with ``adapt_keys`` (which fast-weights receive the online update)
    the canvas methods are recovered as:

    ===========  =====  =====  =====  ==================  ==================
    method       alpha  beta0  beta2  adapt_keys          notes
    ===========  =====  =====  =====  ==================  ==================
    M1 (raw)     1      0      0      ("F",)              decode H frozen to 0 -> innovation = y -> raw K y injection
    M2 (fixed H) 1      0      0      ("F",)              H, K frozen random
    M3 (bridge)  1      ~0.05  0      ("F", "H")          dominant latent + minimal observation anchor
    M4 (obs)     0      1      0      ("F", "H", "K")     parent default
    ===========  =====  =====  =====  ==================  ==================

    ``weight_decay`` applies a decoupled per-update radial shrink
    ``lambda -> (1 - step_size*weight_decay) lambda`` (the linear analog of the
    ``rate(eps)`` penalty that section 6.1 requires alongside the a-posteriori
    term), mirroring the same option in the eigenphase script.
    """

    @dataclass
    class Config(RnnInContextPredictor.Config):
        alpha: float = 0.0                                  # latent self-distillation weight
        beta0: float = 1.0                                  # a-priori observation-prediction weight
        beta2: float = 0.0                                  # a-posteriori observation-prediction weight
        adapt_keys: tuple[str, ...] = ("F", "H", "K")       # fast-weights that receive the online update
        weight_decay: float = 0.0                           # decoupled radial shrink (linear rate-penalty analog)
        sd_horizon: int = 1                                 # number of autonomous self-prediction steps n (1 == today's single-step SD)
        keep_launch: bool = True                            # whether non-root launch states carry gradient (root launch is always detached)

    def __init__(self, modelArgs: "RnnSelfDistillPredictor.Config", **kwargs: Any):
        RnnInContextPredictor.__init__(self, modelArgs, **kwargs)
        self.alpha = modelArgs.alpha
        self.beta0 = modelArgs.beta0
        self.beta2 = modelArgs.beta2
        self.adapt_keys = tuple(modelArgs.adapt_keys)
        self.weight_decay = modelArgs.weight_decay
        self.sd_horizon = modelArgs.sd_horizon
        self.keep_launch = modelArgs.keep_launch

    def _window_loss(self,
                     theta: dict[str, torch.Tensor],
                     s_start: torch.Tensor,                  # [S_D]
                     win_actions: dict[str, torch.Tensor],   # [w x I_D]
                     win_obs: torch.Tensor,                  # [w x O_D]
    ) -> torch.Tensor:                                       # scalar
        """Window-averaged weighted objective (see class docstring).

        The latent self-distillation term is an n-step ladder over horizons
        ``k = 1..sd_horizon``: each target ``sg(x_j^+)`` is predicted by launching
        the a-posteriori state ``k`` steps back, ``x_{j-k}^+``, forward by ``F^k``
        with no innovation. Launch points come from this same rolled window; a
        horizon that reaches past the oldest buffered state clamps to the detached
        root ``s_start`` (so per target only the distinct horizons
        ``k = 1..min(sd_horizon, j+1)`` contribute). At ``sd_horizon=1,
        keep_launch=True`` this is exactly the single-step term ``F x_{j-1}^+``.
        """
        s = s_start
        total = win_obs.new_zeros(())
        w = win_obs.shape[0]
        H_t, F_t = theta["H"], theta["F"]
        # Launch buffer for the SD ladder: ``posts[0]`` is the detached root
        # ``s_start`` (logical launch index -1), ``posts[j + 1]`` is the
        # a-posteriori state ``x_j^+`` of window step ``j``.
        posts = [s_start.detach()]
        for j in range(w):
            action_j = {k: v[j] for k, v in win_actions.items()}
            obs_j = win_obs[j]
            out = torch.func.functional_call(self.cell, theta, (s, action_j, obs_j))
            y_hat = out["environment", "observation"]               # H F x_{t-1}^+
            s_post = out["environment", "state"]                    # x_t^+
            term = win_obs.new_zeros(())
            if self.beta0 != 0.0:
                term = term + self.beta0 * 0.5 * (y_hat - obs_j).pow(2).sum(-1)
            if self.beta2 != 0.0:
                y_post = s_post @ H_t.mT
                term = term + self.beta2 * 0.5 * (y_post - obs_j).pow(2).sum(-1)
            if self.alpha != 0.0:
                # n-step ladder onto the stop-grad target sg(x_j^+). Horizons past
                # the buffered root all collapse to the identical root term, so we
                # iterate only the distinct horizons k = 1..min(sd_horizon, j + 1).
                target = s_post.detach()
                for k in range(1, min(self.sd_horizon, j + 1) + 1):
                    li = j - k                                      # -1 selects the detached root
                    launch = posts[li + 1]
                    if li < 0 or not self.keep_launch:
                        launch = launch.detach()
                    # F^k x_{j-k}^+ built iteratively (vmap/functorch-safe; avoids
                    # a matrix_power batching rule). depth == k here.
                    pred = launch
                    for _ in range(k):
                        pred = pred @ F_t.mT
                    term = term + self.alpha * 0.5 * (target - pred).pow(2).sum(-1)
            total = total + term
            posts.append(s_post)
            s = s_post
        return total / w

    def _optimizer_step(self,
                        theta: dict[str, torch.Tensor],
                        grads: dict[str, torch.Tensor],
                        t: int,
    ) -> dict[str, torch.Tensor]:
        """SGD on the ``adapt_keys`` only (others frozen), with optional decoupled
        weight decay applied to the updated weights. Uses the (possibly decaying)
        online rate ``_step_size_at(t)``."""
        eta_t = self._step_size_at(t)
        updated: dict[str, torch.Tensor] = {}
        for k, v in theta.items():
            if k in self.adapt_keys:
                step = v - eta_t * grads[k]
                if self.weight_decay != 0.0:
                    step = step - eta_t * self.weight_decay * v
                updated[k] = step
            else:
                updated[k] = v
        return updated


class RnnSelfDistillTTTPredictor(RnnPredictor):
    """Static-forward variant of the self-distillation TTT filter, adapted by the
    :class:`~kf_rnn.infrastructure.experiment.training.SelfDistillTTTStage`.

    Unlike :class:`RnnSelfDistillPredictor` (which folds the online adaptation into
    ``forward`` via a fast-weight scan), this model's ``forward`` is the inherited
    static Kalman/IIR filter (:meth:`SequentialPredictor.forward_with_initial`) run
    with the model's *current* parameters. The online self-distillation adaptation
    lives entirely in the training stage: each stage step is one online gradient
    step on a shadow ``(F, H, K)`` iterate, and the reported parameters here are the
    Polyak tail-average of that iterate (see the stage). This inverts the control
    flow so the parameter trajectory, per-step analytical error, and Polyak
    averaging are all first-class, rather than internals hidden inside ``forward``.

    The weighted section-6 objective and the ``(alpha, beta0, beta2, adapt_keys,
    weight_decay, sd_horizon, keep_launch)`` semantics are identical to
    :class:`RnnSelfDistillPredictor`; :meth:`window_loss` is the same math written
    as plain batched matmuls over the ``[N x E x ...]`` ensemble dims (no
    ``functional_call``), so a single ``autograd.grad`` yields per-member gradients.
    Per-method initialization tweaks (M1's ``H = 0``; a random frozen ``K`` when
    ``K`` is not adapted) are applied in ``__init__`` so a method is fully
    specified by its Config.
    """

    @dataclass
    class Config(RnnPredictor.Config):
        # Online-adaptation schedule (mirrors RnnInContextPredictor.Config).
        n_steps: int = 1
        step_size: float = 1.0
        window: int = 1
        step_decay: float = 0.0          # exponent ``a`` in ``eta_t = step_size*(t+1)^(-a)``
        polyak_burnin: float = -1.0      # tail-average start as a fraction of L; < 0 => raw iterate
        # Self-distillation objective (mirrors RnnSelfDistillPredictor.Config).
        alpha: float = 0.0
        beta0: float = 1.0
        beta2: float = 0.0
        adapt_keys: tuple[str, ...] = ("F", "H", "K")
        weight_decay: float = 0.0
        sd_horizon: int = 1
        keep_launch: bool = True
        # Per-method initialization tweaks.
        h_init: str = "random"           # "zero" -> H := 0 (M1 raw-injection)
        frozen_k_random: bool = True     # when "K" not adapted, init K to a random gain

    def __init__(self, modelArgs: "RnnSelfDistillTTTPredictor.Config", **kwargs: Any):
        RnnPredictor.__init__(self, modelArgs, **kwargs)
        self.n_steps = modelArgs.n_steps
        self.step_size = modelArgs.step_size
        self.window = modelArgs.window
        self.step_decay = modelArgs.step_decay
        self.polyak_burnin = modelArgs.polyak_burnin
        self.alpha = modelArgs.alpha
        self.beta0 = modelArgs.beta0
        self.beta2 = modelArgs.beta2
        self.adapt_keys = tuple(modelArgs.adapt_keys)
        self.weight_decay = modelArgs.weight_decay
        self.sd_horizon = modelArgs.sd_horizon
        self.keep_launch = modelArgs.keep_launch

        with torch.no_grad():
            if modelArgs.frozen_k_random and "K" not in self.adapt_keys:
                self.K.copy_(torch.randn((self.S_D, self.O_D)) / (self.O_D ** 0.5))
            if modelArgs.h_init == "zero":
                self.H.zero_()

    def step_size_at(self, t: int) -> float:
        """Online learning rate at global step ``t``: ``step_size * (t+1)^(-step_decay)``."""
        return self.step_size * (t + 1) ** (-self.step_decay)

    @staticmethod
    def kalman_posterior(theta: dict[str, torch.Tensor],
                         s: torch.Tensor,            # [B... x S_D]
                         obs: torch.Tensor,          # [B... x O_D]
    ) -> torch.Tensor:                               # [B... x S_D]
        """One static Kalman step: posterior ``x_t^+`` from the previous posterior
        ``s`` and the observation, using the (batched) filter tensors in ``theta``."""
        F, H, K = theta["F"], theta["H"], theta["K"]
        s_prior = s @ F.mT                                          # x_t^- = F x_{t-1}^+
        y_hat = s_prior @ H.mT                                      # H F x_{t-1}^+
        return s_prior + (obs - y_hat) @ K.mT                       # x_t^+

    def window_loss(self,
                    theta: dict[str, torch.Tensor],
                    s_start: torch.Tensor,           # [B... x S_D]
                    win_obs: torch.Tensor,           # [B... x w x O_D]
    ) -> torch.Tensor:                               # [B...]
        """Window-averaged weighted section-6 objective (see
        :meth:`RnnSelfDistillPredictor._window_loss` for the derivation).

        Batched over arbitrary leading dims ``B...`` (the ``[N x E x M]`` ensemble x
        trace batch). The launch buffer / n-step ladder / ``keep_launch`` semantics
        are identical to the fast-weight version; only the propagation is the plain
        static Kalman step :meth:`kalman_posterior` instead of ``functional_call``.
        """
        F, H = theta["F"], theta["H"]
        s = s_start
        w = win_obs.shape[-2]
        total = win_obs.new_zeros(s_start.shape[:-1])               # [B...]
        posts = [s_start.detach()]      # posts[0] is the detached root (launch index -1)
        for j in range(w):
            obs_j = win_obs[..., j, :]
            s_prior = s @ F.mT
            y_hat = s_prior @ H.mT
            s_post = s_prior + (obs_j - y_hat) @ theta["K"].mT
            term = win_obs.new_zeros(s_start.shape[:-1])
            if self.beta0 != 0.0:
                term = term + self.beta0 * 0.5 * (y_hat - obs_j).pow(2).sum(-1)
            if self.beta2 != 0.0:
                y_post = s_post @ H.mT
                term = term + self.beta2 * 0.5 * (y_post - obs_j).pow(2).sum(-1)
            if self.alpha != 0.0:
                target = s_post.detach()
                for k in range(1, min(self.sd_horizon, j + 1) + 1):
                    li = j - k                                      # -1 selects the detached root
                    launch = posts[li + 1]
                    if li < 0 or not self.keep_launch:
                        launch = launch.detach()
                    pred = launch
                    for _ in range(k):
                        pred = pred @ F.mT
                    term = term + self.alpha * 0.5 * (target - pred).pow(2).sum(-1)
            total = total + term
            posts.append(s_post)
            s = s_post
        return total / w

    def training_recipe(self) -> Sequence[str]:
        return ["self_distill_ttt"]
