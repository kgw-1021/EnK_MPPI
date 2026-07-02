import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp

from quadjax import controllers
from quadjax.dynamics import utils


# ------------------------------ constraints ------------------------------ #
# Constraint functions for the covariance-driven adaptive weighting.
# Each returns a per-sample margin g (shape (N,)); g > 0 means violated.
# They are evaluated on the rolled (next) state + action over the horizon and
# reduced by max, so g = worst violation margin over the plan.

def tilt_constraint(limit_deg: float = 20.0):
    """TOTAL attitude-deviation constraint (roll+pitch+YAW combined).

    Uses quat_w = cos(theta_total/2), so it is violated when the total rotation
    from identity exceeds 2*limit_deg. NOTE: this couples all of roll, pitch AND
    yaw -- in a tracking regime where roll/pitch stay small it is dominated by
    yaw drift, so it effectively constrains heading, not tilt-from-vertical. Use
    `tilt_from_vertical_constraint` for a true tilt (roll/pitch only) constraint.
    """
    cos_lim = float(jnp.cos(jnp.deg2rad(limit_deg)))

    def g(state, action, params):
        return cos_lim - state.quat[:, 3]

    return g


def tilt_from_vertical_constraint(limit_deg: float = 20.0):
    """TRUE tilt constraint: angle between body-z and world-z (roll/pitch only).

    Rzz = 1 - 2*(qx^2 + qy^2) = cos(tilt_from_vertical) is independent of yaw
    (qz). Violated when the drone tilts more than limit_deg from upright. This is
    the physically meaningful "thrust direction / not tipping over" constraint;
    since roll/pitch serve translation, it typically couples to pos/vel (like the
    thrust constraint) rather than to yaw.
    """
    cos_lim = float(jnp.cos(jnp.deg2rad(limit_deg)))

    def g(state, action, params):
        qx, qy = state.quat[:, 0], state.quat[:, 1]
        Rzz = 1.0 - 2.0 * (qx**2 + qy**2)  # = cos(tilt from vertical), yaw-free
        return cos_lim - Rzz

    return g


def thrust_constraint(frac: float = 0.9):
    """Thrust-saturation constraint: thrust must stay under frac * max_thrust."""

    def g(state, action, params):
        thrust = (action[:, 0] + 1.0) / 2.0 * params.max_thrust
        return thrust - frac * params.max_thrust

    return g


@struct.dataclass
class EKSParams:
    gamma_mean: float  # mean of gamma
    gamma_sigma: float  # std of gamma
    discount: float  # discount factor
    sample_sigma: float  # std of sampling

    a_mean: jnp.ndarray  # mean of action
    a_cov: jnp.ndarray  # covariance matrix of action

    weights: jnp.ndarray  # (n_terms,) cost-term weights (same as the other methods)

    # leaky memory of per-term "blame" for covariance-driven adaptive weighting
    # (unused unless the controller is built with adapt=True)
    S: jnp.ndarray = struct.field(default_factory=lambda: jnp.zeros(3))


class EKSController(controllers.BaseController):
    """Ensemble Kalman Sampler controller.

    Identical to MPPI in parameters, isotropic sampling, rollout, and the
    underlying (reward-based) cost -- the ONLY difference is the distribution
    update: MPPI's softmax weighting + weighted mean is replaced by the EKS
    transport. Each inner iteration re-rolls out the transported ensemble and
    applies

        K = C^UG (C^GG + reg I)^-1,   V <- V - K (G - y),   then inflate,

    where V (d, N) is the ensemble of flattened action samples and G is the
    observation built from the cost terms (see `mode`). The posterior ensemble
    mean's first control is returned.

    `mode` controls how much cost *information* the transport observes -- all
    three modes sum to the exact same discounted scalar cost (with the same
    weights as MPPI/CoVO), only the resolution differs:
        "time" : G (H, N)      -- per-timestep cost (terms collapsed)
        "type" : G (n_terms, N)-- per-cost-term (time collapsed)
        "both" : G (H*nt, N)   -- per-timestep AND per-term (finest)
    This lets us measure how EKS performance scales with observation rank.
    """

    def __init__(
        self,
        env,
        control_params,
        N: int,
        H: int,
        lam: float,
        mode: str = "time",
        target: str = "min",
        n_inner: int = 5,
        alpha: float = 10.0,
        inflate: float = 1.0,
        whiten: bool = False,
        terms_fn=None,
        # --- covariance-driven adaptive cost weighting (optional) ---
        adapt: bool = False,
        constraint_fns=None,
        blame_map: str = "signed",
        eta: float = 3.0,
        gamma_adapt: float = 0.8,
        tau: float = 0.03,
        weight_floor: float = 0.15,
        snr_kappa: float = 1.5,
        ucb_z: float = 1.64,
        n_subbatch: int = 4,
    ) -> None:
        super().__init__(env, control_params)
        self.N = N  # NOTE: N is the number of samples, set here as a static number
        self.H = H
        self.lam = lam  # kept for interface parity with MPPI/CoVO (unused by EKS)
        assert mode in ("scalar", "time", "type", "both"), f"unknown EKS mode: {mode}"
        self.mode = mode
        # residual target y in V <- V - K (G - y):
        #   "min"  -> per-row best (lowest-cost) ensemble member  (y = 1.0 * row-min)
        #   "zero" -> absolute zero cost (perfect tracking)        (y = 0)
        #   float  -> y = target * row-min  (probe feasibility/extrapolation)
        if isinstance(target, str):
            assert target in ("min", "zero"), f"unknown EKS target: {target}"
            self.target_scale = 1.0 if target == "min" else 0.0
        else:
            self.target_scale = float(target)
        self.target = target
        self.n_inner = n_inner  # EKS transport iterations
        self.alpha = alpha  # Tikhonov regularization scale (relative)
        self.inflate = inflate  # ensemble inflation factor per inner iteration
        # per-row whitening: normalize each observation row by its own std
        # (covariance -> correlation) so the single scalar reg applies uniformly
        # across rows of very different scale (e.g. weighted pos vs vel vs yaw).
        self.whiten = whiten
        self.action_dim = env.action_dim
        # per-state cost-term function (unweighted); defaults to penyaw tracking
        self.terms_fn = terms_fn if terms_fn is not None else utils.tracking_penyaw_terms_fn
        self.n_terms = control_params.weights.shape[0]
        # number of observation rows G has, given the mode (static)
        self.p = {"scalar": 1, "time": H, "type": self.n_terms, "both": H * self.n_terms}[mode]

        # --- covariance-driven adaptive cost weighting ---
        # Detects, from the ensemble cross-covariance, which cost terms conflict
        # with (or protect) each user constraint, and adaptively re-weights the
        # cost. Robustified against finite-sample noise ("Direction C"):
        #   - sub-batch SNR shrinkage of the alignment a_i (noisy -> neutral)
        #   - UCB (pessimistic) violation probability
        #   - weight floor (no term is ever fully abandoned)
        # blame_map: "signed" (conflict down / align up / decoupled 0, recommended
        #   for physical constraints), "relu" (only penalize conflict, conservative),
        #   or "base" ((1+a)/2, the original -- prone to cross-contamination).
        self.adapt = adapt
        assert blame_map in ("base", "relu", "signed"), f"unknown blame_map: {blame_map}"
        self.blame_map = blame_map
        self.eta = eta  # adaptation strength (softmax temperature on blame)
        self.gamma_adapt = gamma_adapt  # leaky-memory decay
        self.tau = tau  # sigmoid temperature for the soft violation probability
        self.weight_floor = weight_floor  # min weight = weight_floor * w0
        self.snr_kappa = snr_kappa  # SNR at which alignment is half-trusted
        self.ucb_z = ucb_z  # z for the violation-probability upper bound
        self.n_subbatch = n_subbatch  # sub-batches for the SNR estimate
        self.constraint_fns = list(constraint_fns) if constraint_fns else []
        self.n_constraints = len(self.constraint_fns)
        # anchor weights (prior the adaptation reverts to) and total budget
        self.w0 = jnp.asarray(control_params.weights)
        self.Wbud = float(jnp.sum(self.w0))
        if self.adapt:
            assert self.n_constraints > 0, "adapt=True requires constraint_fns"
            assert self.N % self.n_subbatch == 0, "N must be divisible by n_subbatch"

    def _rollout_terms(self, env_state, env_params, a_sampled, step_key):
        """Roll out N action sequences; return per-step cost terms and poses.

        Returns (terms, poses) with terms (H, N, n_terms) the UNWEIGHTED cost
        terms of each rolled state, poses (H, N, 3) for debug stats.
        """

        def rollout_fn(carry, action):
            state, params, terms_before, done_before = carry
            # terms on the pre-step state (the same state reward_fn uses)
            terms = jax.vmap(lambda s: self.terms_fn(s))(state)  # (N, n_terms)
            _, next_state, _, done, _ = jax.vmap(
                lambda s, a, p: self.env.step_env(step_key, s, a, p, True)
            )(state, action, params)
            terms = jnp.where(done_before[:, None], terms_before, terms)
            return (next_state, params, terms, done | done_before), (
                terms,
                next_state.pos,
            )

        state_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_state
        )
        env_params_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_params
        )
        done_repeat = jnp.full(self.N, False)
        terms_repeat = jnp.zeros((self.N, self.n_terms))

        _, (terms, poses) = lax.scan(
            rollout_fn,
            (state_repeat, env_params_repeat, terms_repeat, done_repeat),
            a_sampled.transpose(1, 0, 2),
            length=self.H,
        )
        return terms, poses  # (H, N, n_terms), (H, N, 3)

    def _build_observation(self, terms, weights, disc):
        """Map per-step unweighted terms (H, N, nt) to the observation G (p, N).

        In every mode sum over all rows of G equals the discounted scalar cost
        sum_h disc_h * sum_k w_k * term_k(h); only the resolution differs.
        """
        wterms = terms * weights[None, None, :]  # (H, N, nt)
        if self.mode == "scalar":
            # single total discounted cost (rank-1 observation): (1, N)
            return (wterms.sum(axis=2) * disc[:, None]).sum(axis=0, keepdims=True)
        if self.mode == "time":
            # per-timestep cost (terms collapsed): (H, N)
            return (wterms.sum(axis=2)) * disc[:, None]
        elif self.mode == "type":
            # per-term cost (time collapsed, discounted): (nt, N)
            return (wterms * disc[:, None, None]).sum(axis=0).T
        else:  # "both"
            # per-timestep and per-term: (H*nt, N)
            dwterms = wterms * disc[:, None, None]  # (H, N, nt)
            return dwterms.transpose(0, 2, 1).reshape(self.H * self.n_terms, self.N)

    def _apply_blame_map(self, a):
        if self.blame_map == "base":
            return (1.0 + a) / 2.0  # decoupled -> 0.5 (cross-contamination prone)
        elif self.blame_map == "relu":
            return jnp.maximum(0.0, a)  # only conflict penalized (conservative)
        else:  # "signed"
            return a  # conflict>0 down, align<0 up, decoupled 0

    def _adapt_rollout(self, env_state, env_params, a_sampled, key):
        """Roll out the ensemble; return per-step cost terms (H,N,nt) and the
        per-sample constraint margins g (n_constraints, N) = worst over horizon."""

        def rollout_fn(carry, action):
            state, params = carry
            terms = jax.vmap(lambda s: self.terms_fn(s))(state)
            _, next_state, _, _, _ = jax.vmap(
                lambda s, a, p: self.env.step_env(key, s, a, p, True)
            )(state, action, params)
            cons = jnp.stack(
                [cf(next_state, action, params) for cf in self.constraint_fns], axis=0
            )  # (n_constraints, N)
            return (next_state, params), (terms, cons)

        state_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_state
        )
        params_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_params
        )
        _, (terms, cons) = lax.scan(
            rollout_fn,
            (state_repeat, params_repeat),
            a_sampled.transpose(1, 0, 2),
            length=self.H,
        )
        return terms, cons.max(axis=0)  # (H,N,nt), (n_constraints, N)

    def _adapt_weights(self, env_state, env_params, control_params, key):
        """One outer iteration of covariance-driven adaptive weighting.

        Returns (weights, S): the new per-term weights and updated leaky memory.
        """
        # sample an ensemble around the current (warm-started) mean
        key, sk = jax.random.split(key)
        act_keys = jax.random.split(sk, self.N)

        def single_sample(k, mean, cov):
            ks = jax.random.split(k, self.H)
            return jax.vmap(
                lambda k, m, c: jax.random.multivariate_normal(k, m, c)
            )(ks, mean, cov)

        a_s = jax.vmap(single_sample, in_axes=(0, None, None))(
            act_keys, control_params.a_mean, control_params.a_cov
        )
        a_s = jnp.clip(a_s, -1.0, 1.0)  # (N, H, action_dim)

        terms, gmar = self._adapt_rollout(env_state, env_params, a_s, key)
        Jtot = terms.sum(axis=0)  # (N, n_terms) unweighted cost-term totals
        U = a_s.reshape(self.N, -1)  # (N, d)

        # UCB (pessimistic) violation probability per constraint
        sig = jax.nn.sigmoid(gmar / self.tau)  # (n_constraints, N)
        pv = sig.mean(1) + self.ucb_z * sig.std(1) / jnp.sqrt(self.N)  # (n_constraints,)

        # sub-batch estimate of the control-cost/constraint alignment a_{k,i}
        B, nb = self.n_subbatch, self.N // self.n_subbatch
        Ub = U[: B * nb].reshape(B, nb, -1)
        Jb = Jtot[: B * nb].reshape(B, nb, self.n_terms)
        gb = gmar[:, : B * nb].reshape(self.n_constraints, B, nb).transpose(1, 0, 2)

        def per_batch(Ub_, Jb_, gb_):  # (nb,d),(nb,nt),(n_constraints,nb)
            Uc = Ub_ - Ub_.mean(0)
            PJ = Uc.T @ (Jb_ - Jb_.mean(0)) / nb  # (d, nt)
            Pg = Uc.T @ (gb_.T - gb_.T.mean(0)) / nb  # (d, n_constraints)
            PJn = PJ / (jnp.linalg.norm(PJ, axis=0) + 1e-9)
            Pgn = Pg / (jnp.linalg.norm(Pg, axis=0) + 1e-9)
            return -(Pgn.T @ PJn)  # (n_constraints, nt)

        ab = jax.vmap(per_batch)(Ub, Jb, gb)  # (B, n_constraints, nt)
        a_mean = ab.mean(0)  # (n_constraints, nt)
        a_se = ab.std(0) / jnp.sqrt(B)
        snr = jnp.abs(a_mean) / (a_se + 1e-9)
        reliab = snr / (snr + self.snr_kappa)  # finite-sample shrinkage
        a_used = a_mean * reliab

        # blame = sum over constraints of p_viol * mapped(alignment)
        c = (pv[:, None] * self._apply_blame_map(a_used)).sum(0)  # (n_terms,)
        S = self.gamma_adapt * control_params.S + c

        w = self.w0 * jnp.exp(-self.eta * S)
        w = self.Wbud * w / w.sum()
        w = jnp.maximum(w, self.weight_floor * self.w0)  # no term abandoned
        w = self.Wbud * w / w.sum()
        return w, S

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs: jnp.ndarray,
        env_state,
        env_params,
        rng_act: chex.PRNGKey,
        control_params: EKSParams,
        info,
    ) -> jnp.ndarray:
        # inject noise to env_state elements
        env_state = info["noisy_state"]

        # covariance-driven adaptive cost weighting (optional; before the shift so
        # blame is measured on the current warm-started ensemble)
        if self.adapt:
            rng_act, adapt_key = jax.random.split(rng_act)
            new_w, new_S = self._adapt_weights(
                env_state, env_params, control_params, adapt_key
            )
            control_params = control_params.replace(weights=new_w, S=new_S)

        # shift operator
        a_mean_old = control_params.a_mean
        a_cov_old = control_params.a_cov

        control_params = control_params.replace(
            a_mean=jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]]),
            a_cov=jnp.concatenate([a_cov_old[1:], a_cov_old[-1:]]),
        )

        # sample action with mean and covariance, repeat for N times (N, H, action_dim)
        rng_act, act_key = jax.random.split(rng_act)
        act_keys = jax.random.split(act_key, self.N)

        def single_sample(key, traj_mean, traj_cov):
            keys = jax.random.split(key, self.H)
            return jax.vmap(
                lambda key, mean, cov: jax.random.multivariate_normal(key, mean, cov)
            )(keys, traj_mean, traj_cov)

        a_sampled = jax.vmap(single_sample, in_axes=(0, None, None))(
            act_keys, control_params.a_mean, control_params.a_cov
        )
        a_sampled = jnp.clip(a_sampled, -1.0, 1.0)  # (N, H, action_dim)

        # per-step discount weights (so observation rows sum to the discounted cost)
        disc = jnp.power(control_params.discount, jnp.arange(self.H))  # (H,)
        weights = control_params.weights

        # ensemble of flattened action samples: V (d, N), d = H * action_dim
        V = a_sampled.reshape(self.N, -1).T

        # === EKS transport (replaces MPPI softmax weighting + weighted mean) ===
        def inner(carry, _):
            V, key = carry
            key, step_key = jax.random.split(key)
            a_cur = jnp.clip(V.T.reshape(self.N, self.H, self.action_dim), -1.0, 1.0)
            terms, poses = self._rollout_terms(
                env_state, env_params, a_cur, step_key
            )
            G = self._build_observation(terms, weights, disc)  # (p, N)
            # target y = target_scale * per-row min (1.0 -> best member, 0.0 -> zero cost)
            y = self.target_scale * jnp.min(G, axis=1, keepdims=True)  # (p, 1)

            Vc = V - V.mean(axis=1, keepdims=True)
            Gc = G - G.mean(axis=1, keepdims=True)
            resid = G - y  # (p, N)
            if self.whiten:
                # per-row std -> whiten Gc and the residual so every row has unit
                # variance (Cgg becomes a correlation matrix); the single reg then
                # applies uniformly and low-weight rows are not regularized away.
                s = jnp.sqrt(jnp.mean(Gc**2, axis=1, keepdims=True) + 1e-12)  # (p,1)
                Gc = Gc / s
                resid = resid / s
            Cug = (Vc @ Gc.T) / self.N  # (d, p)
            Cgg = (Gc @ Gc.T) / self.N  # (p, p)
            # relative Tikhonov regularization (scale-invariant in the cost units);
            # larger alpha -> smaller reg -> closer to the full Gauss-Newton step
            reg = jnp.trace(Cgg) / self.p / self.alpha + 1e-8
            K = Cug @ jnp.linalg.inv(Cgg + reg * jnp.eye(self.p))  # (d, p)
            V = V - K @ resid  # (d, N)
            # multiplicative inflation around the new mean
            vbar = V.mean(axis=1, keepdims=True)
            V = vbar + (V - vbar) * self.inflate
            return (V, key), poses

        (V, _), poses_seq = lax.scan(inner, (V, rng_act), None, length=self.n_inner)

        # posterior ensemble mean, blended with gamma_mean (identical to MPPI)
        a_mean_eks = V.mean(axis=1).reshape(self.H, self.action_dim)
        a_mean = a_mean_eks * control_params.gamma_mean + control_params.a_mean * (
            1 - control_params.gamma_mean
        )
        control_params = control_params.replace(a_mean=a_mean)

        # get action
        u = control_params.a_mean[0]

        # debug values (from the final transport iteration)
        poses = poses_seq[-1]
        info = {"pos_mean": jnp.mean(poses, axis=1), "pos_std": jnp.std(poses, axis=1)}

        return u, control_params, info
