# Minimal convergent online-filter proposal

A proposal for the *smallest-hyperparameter* online adaptive filter that is
**guaranteed to reach the irreducible Kalman floor** in **constant memory** on a
**single trajectory**, for the linear LQE toy case. This is the constructive
follow-up to the negative results in
[`self_distillation_study_manifest.md`](self_distillation_study_manifest.md)
section 6: the M1-M4 / self-distillation losses, as currently run, plateau ~0.2-3%
above the floor and are matched by a length-2 FIR, while online least squares with
`R>=4` reaches the floor to ~1e-6. The question this answers is *what is the
minimal change that closes that gap, and which knobs are theoretically free*.

The contents:

0. Problem statement and success criterion.
1. The minimal method (one objective, one estimator).
2. Why it converges to zero excess (assumptions + proof sketch).
3. Hyperparameter inventory: **load-bearing** vs **free**.
4. Knobs that can be changed while preserving the zero-convergence guarantee.
5. Relevant works.
6. Mapping onto the current code.
7. Minimal validation ladder.

---

## 0. Problem statement and success criterion

- **System.** Stable LTI state-space model, partially observed, driven by
  stationary (Gaussian) process/observation noise -- the `sd6_od2` regime of the
  study (`S_D=6`, `O_D=2`).
- **Setting.** *One* trajectory, observed online; **constant memory and
  per-step compute** (no growing buffer, no storing the history).
- **Filter class.** A recurrent (IIR) linear filter `(F, H, K)` -- the Kalman
  structure -- so the optimum is **in the class** (no FIR truncation tail).
- **Success criterion.** Asymptotic **excess over the irreducible floor**
  `(err - floor)/floor -> 0` as the horizon `n -> infinity`, in probability
  (a.s. under the stronger assumptions below). This is the exact metric in
  §5.4 / §6 of the manifest (`filter_analytical_error`).

The conceptual resolution that licenses this (see the chat that produced this
doc): bounded memory != truncated memory. For linear-Gaussian filtering the whole
past is summarised by a **finite-dimensional sufficient statistic** (the filter
state), so a constant-memory recursion is *exactly* optimal; and a single
**ergodic** trace makes the **time-average** objective converge to the
**ensemble** objective the Kalman filter minimises -- no ensemble of traces is
needed.

---

## 1. The minimal method

One objective and one estimator. The two presentations below are equivalent in
the limit; pick by implementation taste.

### 1.1 Objective: the a-priori one-step prediction error (and nothing else)

$$
J_n(\theta) \;=\; \frac{1}{n}\sum_{t=1}^{n} \tfrac12\,\big\|\,H F\hat x_{t-1}^+ - y_t\,\big\|^2,
\qquad \theta=(F,H,K).
$$

This is the **prediction-error method (PEM)** loss. Its population minimiser
(expectation over the stationary law) is the Kalman/Wiener filter. This is
exactly the existing `beta0` term; **`beta2 = 0`** (the a-posteriori term has a
target-leakage bias -- the manifest §6.1 negative result), and **`alpha = 0`**
(latent self-distillation changes the minimiser unless annealed; keep it only as
an *optional* transient regulariser, §4). So the objective has **zero loss-shape
hyperparameters** -- it is a single squared one-step residual.

### 1.2 Estimator (primary): recursive Gauss-Newton PEM (self-tuning gain)

Carry, in the online recursion, the **output sensitivity**
$\psi_t = \mathrm{d}\hat y_t/\mathrm{d}\theta$ (a *gradient filter* -- the linear
state-space RTRL recursion; Williams & Zipser 1989, Ljung & Söderström 1983),
the running curvature $A_t$, and the parameter. With the predictor in innovations
form and closed loop $A_{\mathrm{cl}}=(I-KH)F$,

$$
\hat y_t = HF\hat x_{t-1},\qquad \varepsilon_t = y_t-\hat y_t,\qquad
\hat x_t = A_{\mathrm{cl}}\hat x_{t-1}+K y_t,
$$

the sensitivity and update recursions are

$$
S_t = \underbrace{D_t(\theta_t;\hat x_{t-1},y_t)}_{\text{explicit }\partial/\partial\theta}
      + A_{\mathrm{cl}}\,S_{t-1},
\qquad
\psi_t = \frac{\partial(HF\hat x_{t-1})}{\partial(H,F)} + HF\,S_{t-1},
$$
$$
A_t = \lambda A_{t-1} + \psi_t\psi_t^\top,\qquad
\theta_t = \theta_{t-1} + A_t^{-1}\psi_t\,\varepsilon_t,
$$

where $S_t=\mathrm{d}\hat x_t/\mathrm{d}\theta$ is a constant-size carry and
$\lambda=1$ for the stationary case. The gain $A_t^{-1}$ is **computed from the
data**, so there is **no learning rate** to tune.

*Definition of $D_t$.* It is the **explicit** Jacobian of the state update
w.r.t. $\theta$ with $\hat x_{t-1}$ held fixed, i.e.
$D_t=\partial\hat x_t/\partial\theta|_{\hat x_{t-1}\text{ const}}$. Differentiating
$\hat x_t = F\hat x_{t-1}+K\varepsilon_t$ gives the three parameter blocks
$$
\mathrm{d}\hat x_t = \underbrace{(I-KH)(\mathrm{d}F)\hat x_{t-1}
- K(\mathrm{d}H)F\hat x_{t-1} + (\mathrm{d}K)\varepsilon_t}_{D_t}
+ \underbrace{A_{\mathrm{cl}}\,\mathrm{d}\hat x_{t-1}}_{A_{\mathrm{cl}}S_{t-1}},
$$
per-entry $\partial\hat x_t/\partial F_{ij}=(A_{\mathrm{cl}})_{:,i}(\hat x_{t-1})_j$,
$\partial\hat x_t/\partial H_{ij}=-K_{:,i}(F\hat x_{t-1})_j$,
$\partial\hat x_t/\partial K_{ij}=e_i(\varepsilon_t)_j$. Every factor is already
on hand at step $t$, so $D_t$ is $O(1)$ to form.

*Where is K?* The explicit $\psi_t$ term is over $(H,F)$ only because the
prediction $\hat y_t=HF\hat x_{t-1}$ contains **no explicit K** -- K acts on the
*state update*, not the prediction, so
$\partial\hat y_t/\partial K|_{\hat x_{t-1}\text{ fixed}}=0$. K's gradient is
nonetheless fully in $\psi_t$, carried by the K-slice of $S_{t-1}$ in the
$HF\,S_{t-1}$ term. It is *born* in the **state** recursion's explicit term
$D_t$, which does see K explicitly:
$\partial\hat x_t/\partial K|_{\hat x_{t-1}\text{ fixed}}
= y_t-HF\hat x_{t-1}=\varepsilon_t$ (from $\hat x_t=A_{\mathrm{cl}}\hat x_{t-1}+Ky_t$).
That K-sensitivity then propagates through $A_{\mathrm{cl}}S_{t-1}$ into the next
prediction -- the carry-form of the "one step before K's gradient appears" fact
in §1.5. This is the recurrent analogue
of the FIR online-LS already in the codebase, which accumulates exactly this kind
of normal-equation sufficient statistic
([`cnn_predictor.py` lines 110-115](../src/kf_rnn/model/convolutional/cnn_predictor.py)).

**State staleness is not a problem (and needs no recomputation).** $\hat x_t$ was
produced by the *moving* sequence $\theta_1,\dots,\theta_{t-1}$, not a single
$\theta$, so there is no fixed $\theta$ to re-differentiate. The RPEM resolution
is to never recompute the state: carry $S_t$ forward, evaluating its Jacobians at
the **current** $\theta_t$. The residual mismatch is benign for two reasons that
hold *exactly in the limit*: (i) the gain vanishes ($A_t^{-1}\!\sim\!1/t$), so
$\theta_t$ varies ever more slowly -- the slow-variation regime where Ljung's
associated-ODE analysis proves the carried $S_t$ converges to the true stationary
sensitivity; and (ii) the carry's transition is the closed-loop contraction
$A_{\mathrm{cl}}$ (guaranteed $\rho\le 1-\epsilon$ by the stability projection,
§1.4), so contributions from stale early $\theta$'s are forgotten geometrically.

### 1.3 Estimator (alternative): averaged SGD (simplest to implement)

Identical gradient $\psi_t\varepsilon_t$ as §1.2 (the **same** sensitivity carry
-- *not* a windowed/detached re-roll; see §1.5), but a scalar gain with
**Polyak-Ruppert tail averaging** (Polyak & Juditsky 1992) as the reported
filter:

$$
\theta_t = \theta_{t-1} + \eta_t\,\psi_t\varepsilon_t,
\qquad
\bar\theta_n = \frac{1}{n-n_0}\sum_{t>n_0}\theta_t .
$$

The *averaged* iterate $\bar\theta_n$ is asymptotically efficient (attains the
same $H^{-1}\Sigma H^{-1}/n$ covariance as recursive Gauss-Newton) for **any**
$\eta_t=\eta_0 t^{-a}$ with $a\in(\tfrac12,1)$. The only difference from §1.2 is
the gain (scalar averaged step vs. data-driven matrix $A_t^{-1}$). This is the
cheapest possible change to the current loop and the recommended first diagnostic
(§7, Tier 0).

### 1.4 The two non-negotiable structural pieces

These are not hyperparameters; they are correctness requirements:

- **Untruncated gradient.** Use the sensitivity recursion (RTRL) -- *not* a
  detached, fixed-length window. A finite window `W` leaves an
  $O(|\lambda_{\mathrm{cl}}|^{W})$ bias on the slow modes (the near-unit poles
  this system needs), which is exactly the current `window=4`, `s_start.detach()`
  defect. The sensitivity carry is constant size, so this is still constant
  memory.
- **Stability projection.** After each update, keep the closed loop a
  contraction: project the spectral radius of $F(I-KH)$ to $\le 1-\epsilon$ for a
  small fixed $\epsilon$. This fixes the §6.3 pathology (`median|lambda(F_hat)|=1.0`,
  filters tipping unstable). $\epsilon$ is a tolerance, not a tuned knob (any
  value that does not exclude the true closed loop works).

### 1.5 Why there is no window, no stop-gradient, and yet a K-gradient

A natural objection to a *windowed* formulation: with a length-1 window and a
detached entering state, $\hat y_t = HF\hat x_{t-1}$ treats $\hat x_{t-1}$ as a
constant, so the loss has **no gradient w.r.t. `K`** (K only entered through how
$\hat x_{t-1}$ was made). The windowed fix would need `window >= 2` (so the
in-window update $\hat x_{t-1}=A_{\mathrm{cl}}\hat x_{t-2}+Ky_{t-1}$ exposes K),
and then re-rolling the `W`-step window every timestep ($O(W)$ recompute) while
still eating an $O(|\lambda_{\mathrm{cl}}|^{W})$ truncation bias.

The sensitivity carry (§1.2) **dissolves this**:

- **No window, no detach, no re-run.** $S_{t-1}$ already holds the full-history
  dependence in constant memory; the gradient is exact at $O(1)$ per step.
- **K's gradient appears automatically.** $\hat y_t=HF\hat x_{t-1}$ has no
  *explicit* `K`, so $\psi_t$'s K-component flows entirely through $S_{t-1}$,
  which accumulates a K-dependence via its explicit term $D$
  ($\partial(Ky-KHF\hat x)/\partial K\sim\varepsilon$). This is the RTRL analogue
  of "need window 2 for K" -- it materialises after **one** carried step, for
  free.

So both §1.2 and §1.3 use this carry; windowed TBPTT is only the *approximate*
fallback (`W>=2`, re-roll each step, $O(|\lambda|^{W})$ bias), and constant-memory
**zero** convergence specifically requires the exact gradient, not the window.

---

## 2. Why it converges to zero excess

**Assumptions.**

- (A1) **Stability:** the data-generating closed loop is a strict contraction
  (stable LTI), so the output is stationary and the transient decays
  geometrically.
- (A2) **Ergodicity / persistent excitation:** the process noise excites all
  modes; the output spectrum is absolutely continuous (no pure tones). Holds for
  a noise-driven stable LTI system.
- (A3) **Identifiability:** the true filter is a (locally) isolated minimiser of
  the population PEM loss, up to the unobservable similarity transform.
- (A4) **Order:** model state dimension $\ge$ the true minimal order.

**Sketch.**

1. *Single trace suffices.* By (A1)-(A2) and the ergodic theorem,
   $J_n(\theta)\to \mathbb E\,\tfrac12\|\hat y-y\|^2$ uniformly on compacts; the
   minimiser of the time-average converges to the minimiser of the ensemble
   objective -- the Wiener/Kalman filter. No ensemble of trajectories is needed
   (the objection that "we only get one infinite trace" is resolved here).
2. *Constant memory suffices.* The filter state is a finite sufficient statistic
   for the infinite past (Kalman 1960); the sensitivity $\psi_t$ is a finite
   sufficient statistic for the exact online gradient (Williams & Zipser 1989).
   Neither grows with $n$.
3. *No asymptotic bias.* With the untruncated gradient (§1.4) and `beta2=alpha=0`
   and ridge $\to 0$, the only stationary point the recursion is attracted to is
   the population minimiser; there is no truncation tail and no target leakage.
4. *No variance floor.* Recursive Gauss-Newton (1.2) is the stochastic-Newton /
   recursive-PEM scheme whose estimate is consistent and asymptotically efficient
   (Ljung & Söderström 1983); equivalently, averaged SGD (1.3) attains the same
   optimal rate (Polyak & Juditsky 1992). A **constant** gain, by contrast,
   converges only to a noise ball of radius $\propto\sqrt{\eta}$ (Dieuleveut,
   Durmus & Bach 2020) -- this is the current plateau.

Conclusion: excess $\to 0$ at the optimal $O(1/n)$ rate, in constant memory, on
one trace.

---

## 3. Hyperparameter inventory

The design goal is that the **asymptote depends on nothing tunable**. Every knob
is classified by whether it can change the *zero-error guarantee* (load-bearing)
or only the *transient/constants* (free).

| knob | role | classification | required value / range for zero excess |
|---|---|---|---|
| loss weights `(beta0, beta2, alpha)` | objective | **load-bearing** | `beta0>0`, **`beta2=0`**, **`alpha=0`** (or annealed `alpha_t -> 0`) |
| gradient truncation `window` | gradient bias | **load-bearing** | exact (RTRL) i.e. `window = infinity`; finite fixed `W` leaves $O(|\lambda|^{W})$ bias |
| ridge / regulariser | conditioning | **load-bearing** | $\to 0$ as $n\to\infty$ |
| model order `S_D` | capacity | **load-bearing (lower bound)** | $\ge$ true minimal order |
| gain type | variance floor | **load-bearing** | vanishing/self-tuning (RGN) or averaged (Polyak); **constant gain fails** |
| stability margin `eps` | feasibility | tolerance | any `eps>0` not excluding the true closed loop |
| base step `eta_0` (averaged-SGD only) | transient | **free** | any `>0` |
| step exponent `a` (averaged-SGD only) | transient | **free** | any `a in (1/2, 1)` |
| averaging burn-in `n_0` | transient | **free** | any `n_0 = o(n)` |
| forgetting factor `lambda` (RGN) | tracking | **free for stationary** | `lambda = 1`; `lambda<1` trades floor for tracking |
| initialization / warm start | basin | **free*** | any point in the true basin (*see §3.1) |
| number of parallel traces `N` | efficiency | **free** | `>=1`; more only improves constants |

### 3.1 The one genuinely hard knob: initialization

The recurrent PEM loss is **nonconvex** (RLS's clean *global* guarantee is
FIR-only). Convergence to the *global* optimum is only guaranteed from within the
true basin. Minimal robust fix: **warm-start** $(F,H,K)$ from a cheap
batch/subspace solve on the first few hundred steps (the codebase already has the
LS machinery), then hand off to the online recursion. This is consistency
infrastructure, not a tuned hyperparameter.

---

## 4. Knobs you can change while keeping the guarantee

These follow from §2-3 and are the "theoretically free" axes (each cite is the
result that licenses it):

- **Base step size `eta_0`** (averaged-SGD): any positive value. With
  Polyak-Ruppert averaging the asymptotic covariance is independent of `eta_0`
  (Polyak & Juditsky 1992); `eta_0` only moves the burn-in.
- **Step decay exponent `a in (1/2, 1)`**: the entire open interval yields the
  same asymptotically efficient averaged estimate. `a<=1/2` loses the averaging
  guarantee; `a>=1` decays too fast (gets stuck).
- **Averaging burn-in `n_0`**: any sublinear schedule (`n_0=o(n)`, e.g. a fixed
  fraction). Affects constants, not the limit.
- **Forgetting factor `lambda` (RGN)**: `lambda=1` for the stationary case gives
  zero excess. Any `lambda<1` introduces a tracking floor `~(1-lambda)` (Ljung &
  Söderström 1983) -- so `lambda` is free *iff* you keep it `=1` (it becomes
  load-bearing the moment you want nonstationary tracking).
- **Model order `S_D`**: any value `>= true order`. Over-parameterisation stays
  consistent under (A3) (extra modes are unidentifiable but harmless); under-
  parameterisation breaks the guarantee.
- **Number of trajectories `N` / mini-batching**: any `N>=1`. More traces reduce
  the variance constant (the autocorrelation-inflated effective sample size) but
  the single-trace asymptote is already zero.
- **Stability margin `eps`**: any small `eps>0` that does not exclude the true
  closed loop.
- **Optional latent SD `alpha_t`**: allowed **only if annealed to 0**
  (`sum alpha_t < infinity` style), in which case it is a vanishing regulariser
  that cannot bias the limit; a fixed `alpha>0` is load-bearing (shifts the
  minimiser).

Conversely, the axes that **cannot** be relaxed: a nonzero `beta2`, a fixed
finite gradient window, a non-vanishing (constant) gain, a non-vanishing ridge,
and an under-sized model order. Each reintroduces a specific, named error term.

---

## 5. Relevant works

**Optimal filtering / sufficient statistics.**
- R. E. Kalman (1960), *A New Approach to Linear Filtering and Prediction
  Problems.* -- the constant-memory optimal recursion; the floor.
- N. Wiener (1949), *Extrapolation, Interpolation, and Smoothing of Stationary
  Time Series.* -- optimal filter from second-order statistics (spectrum).

**Recursive identification / PEM (the primary estimator).**
- L. Ljung & T. Söderström (1983), *Theory and Practice of Recursive
  Identification.* -- recursive PEM, recursive Gauss-Newton, gradient/sensitivity
  filter, forgetting factor.
- L. Ljung (1999), *System Identification: Theory for the User* (2nd ed.).
- P. Van Overschee & B. De Moor (1996), *Subspace Identification for Linear
  Systems.* -- consistent constant-order init from single-trace covariances
  (warm start, §3.1).

**Stochastic approximation, averaging, and the constant-step floor.**
- H. Robbins & S. Monro (1951), *A Stochastic Approximation Method.*
- D. Ruppert (1988); B. Polyak & A. Juditsky (1992), *Acceleration of Stochastic
  Approximation by Averaging.* -- the averaged-SGD efficiency result behind §1.3.
- A. Dieuleveut, A. Durmus & F. Bach (2020), *Bridging the Gap between Constant
  Step Size SGD and Markov Chains.* -- formalises the constant-gain noise ball
  (the current plateau).
- L. Bottou & Y. Le Cun (2005), *On-line Learning for Very Large Datasets*;
  S. Amari (1998), *Natural Gradient Works Efficiently in Learning.* -- second-
  order / natural-gradient view of the self-tuning gain.

**Recursive least squares / adaptive filtering (the FIR analogue in-repo).**
- S. Haykin, *Adaptive Filter Theory*; A. H. Sayed, *Adaptive Filters.*
- B. Widrow & S. Stearns, *Adaptive Signal Processing* (LMS/NLMS context).

**Online RNN gradients (the untruncated-gradient requirement, §1.4).**
- R. Williams & D. Zipser (1989), *A Learning Algorithm for Continually Running
  Fully Recurrent Neural Networks* (RTRL).
- P. Werbos (1990), *Backpropagation Through Time* (and its truncation, TBPTT --
  the bias source we remove).

**Single-trajectory finite-sample LDS / Kalman learning (single-trace theory).**
- M. Hardt, T. Ma & B. Recht (2018), *Gradient Descent Learns Linear Dynamical
  Systems.*
- M. Simchowitz, H. Mania, S. Tu, M. Jordan & B. Recht (2018), *Learning Without
  Mixing.*
- A. Tsiamis & G. Pappas (2019), *Finite Sample Analysis of Stochastic System
  Identification*; and Kalman-filter learning follow-ups.

**Stability-constrained parametrisations (the projection, §1.4).**
- N. Gillis et al., stable LTI parametrisations; orthogonal/antisymmetric and
  Lipschitz RNNs (Chang et al. 2019; Erichson et al. 2021); diagonal-stable SSMs
  (Gu et al., S4, 2022) as practical stable parametrisations.

**Test-time training (the framing of the study).**
- Y. Sun et al. (2020), *Test-Time Training with Self-Supervision*; and the
  online/continual TTT line -- the constant-gain self-supervised update whose
  stationary-LTI limit this proposal corrects.

**Bootstrap target (the optional latent-SD term).**
- R. Sutton (1988), *Learning to Predict by the Methods of Temporal Differences.*

---

## 6. Mapping onto the current code

All changes are localised to the overridable hooks in
[`rnn_ttt.py`](../src/kf_rnn/model/sequential/rnn_ttt.py); propagation and the
scan are untouched.

| piece | current | minimal-method change |
|---|---|---|
| objective | `_window_loss` weighted 3-term | set `beta0=1, beta2=0, alpha=0`; loss becomes the single a-priori residual (no code change, just weights) |
| gradient | `_compute_grads` = truncated `_window_loss` over `window=4`, `s_start.detach()` ([lines 142-148](../src/kf_rnn/model/sequential/rnn_ttt.py)) | replace with a sensitivity (RTRL) carry in the `scan` -- exact online gradient, constant memory |
| gain | `_optimizer_step` = constant-step SGD ([lines 267-282](../src/kf_rnn/model/sequential/rnn_ttt.py)) | (a) add Polyak tail-averaging of `theta`; or (b) recursive Gauss-Newton with carried `A_t` (mirror the normal-equation accumulation in [`cnn_predictor.py` 110-115](../src/kf_rnn/model/convolutional/cnn_predictor.py)) |
| stability | none (`|lambda|=1`, §6.3) | project `rho(F(I-KH)) <= 1-eps` after each update |
| init | `K=0` default | warm-start from a short batch/subspace LS solve |

---

## 7. Minimal validation ladder

Run on the `sd6_od2`, `eps=0.1`, `L=10000` setup of §6 so it is directly
comparable to the existing curves (and to the online-LS / optimal-FIR overlays).

0. **Tier 0 (free diagnostic): Polyak tail-averaging on the existing M4 run.**
   Output `theta_bar` instead of `theta`. *Prediction:* the ~2e-4 M4 plateau
   drops toward the floor. Confirms the variance-floor diagnosis with ~2 lines.
1. **Tier 1: decaying step `eta_t ~ eta_0 t^{-a}`** (with averaging) -- confirms
   the gain axis; check insensitivity to `eta_0`, `a in (1/2,1)`.
2. **Tier 2: untruncated (RTRL) gradient** -- removes the residual slow-pole
   bias; the *converged floor* (not just the rate) should improve.
3. **Tier 3: recursive Gauss-Newton + stability projection + warm start** -- the
   full minimal method; target is the irreducible floor (matching online-LS) in
   an IIR class, i.e. with no FIR truncation tail.

Success at Tier 3 is the headline this proposal is designed to produce: a
constant-memory, single-trace online filter sitting on the irreducible floor.
