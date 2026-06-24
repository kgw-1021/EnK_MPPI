import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp

from quadjax import controllers
from quadjax.dynamics import utils


@struct.dataclass
class EKSParams:
    gamma_mean: float  # mean of gamma
    gamma_sigma: float  # std of gamma
    discount: float  # discount factor
    sample_sigma: float  # std of sampling

    a_mean: jnp.ndarray  # mean of action
    a_cov: jnp.ndarray  # covariance matrix of action

    weights: jnp.ndarray  # (n_terms,) cost-term weights (same as the other methods)


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
        terms_fn=None,
    ) -> None:
        super().__init__(env, control_params)
        self.N = N  # NOTE: N is the number of samples, set here as a static number
        self.H = H
        self.lam = lam  # kept for interface parity with MPPI/CoVO (unused by EKS)
        assert mode in ("time", "type", "both"), f"unknown EKS mode: {mode}"
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
        self.action_dim = env.action_dim
        # per-state cost-term function (unweighted); defaults to penyaw tracking
        self.terms_fn = terms_fn if terms_fn is not None else utils.tracking_penyaw_terms_fn
        self.n_terms = control_params.weights.shape[0]
        # number of observation rows G has, given the mode (static)
        self.p = {"time": H, "type": self.n_terms, "both": H * self.n_terms}[mode]

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
            Cug = (Vc @ Gc.T) / self.N  # (d, p)
            Cgg = (Gc @ Gc.T) / self.N  # (p, p)
            # relative Tikhonov regularization (scale-invariant in the cost units);
            # larger alpha -> smaller reg -> closer to the full Gauss-Newton step
            reg = jnp.trace(Cgg) / self.p / self.alpha + 1e-8
            K = Cug @ jnp.linalg.inv(Cgg + reg * jnp.eye(self.p))  # (d, p)
            V = V - K @ (G - y)  # (d, N)
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
