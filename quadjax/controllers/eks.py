import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp

from quadjax import controllers


@struct.dataclass
class EKSParams:
    """Parameters/state for the Ensemble Kalman Sampler controller.

    Sampling is a fixed isotropic Gaussian around the warm-started mean
    (MPPI-style); the "intelligence" lives in the Gauss-Newton transport, not in
    the sampling distribution. `gw` weights the four residual terms
    [track, vel, smooth, effort].
    """

    a_mean: jnp.ndarray  # (H, action_dim) warm-started control mean
    a_hover: jnp.ndarray  # (action_dim,) effort target (hover action)
    sample_sigma: jnp.ndarray  # (action_dim,) per-dim isotropic sampling scale
    gw: jnp.ndarray  # (4,) residual-term weights [track, vel, smooth, effort]
    a_cov: jnp.ndarray  # (H*action_dim, H*action_dim) posterior covariance (debug)


class EKSController(controllers.BaseController):
    """Ensemble Kalman Sampler transport controller.

    Each step samples an ensemble of control sequences around the warm-started
    mean, then runs `n_inner` Gauss-Newton/Tikhonov Kalman updates that transport
    the ensemble toward the posterior:

        K = C^UG (C^GG + (1/alpha) I)^-1,   V <- V - K G,   then inflate,

    where G stacks the (sqrt-weighted) residual blocks. Residuals are built from
    the rolled-out trajectory (`env.step_env`) plus the actions:
        track  = pos  - pos_tar      (from rollout)
        vel    = vel  - vel_tar      (from rollout)
        smooth = a[h] - a[h-1]       (from actions)
        effort = a[h] - a_hover      (from actions)
    The posterior mean's first control is returned.
    """

    def __init__(
        self,
        env,
        control_params,
        N: int,
        H: int,
        n_inner: int = 10,
        alpha: float = 1.0,
        inflate: float = 1.05,
    ) -> None:
        super().__init__(env, control_params)
        self.N = N  # number of ensemble members (static)
        self.H = H  # horizon (static)
        self.n_inner = n_inner  # Gauss-Newton transport iterations (static)
        self.alpha = alpha  # Tikhonov regularization (1/alpha on the diagonal)
        self.inflate = inflate  # ensemble inflation factor per inner iteration
        self.action_dim = env.action_dim

    def _rollout_residuals(
        self, env_state, env_params, a_sampled: jnp.ndarray, gw: jnp.ndarray,
        a_hover: jnp.ndarray, key: chex.PRNGKey,
    ):
        """Roll out N action sequences and build the residual matrix G (p, N).

        Returns (G, pos_seq) where pos_seq is (H, N, 3) for debug statistics.
        """
        N, H, ad = self.N, self.H, self.action_dim

        def rollout_fn(carry, action):
            state, params, done_before = carry
            _, state, _, done, _ = jax.vmap(
                lambda s, a, p: self.env.step_env(key, s, a, p, True)
            )(state, action, params)
            return (state, params, done | done_before), (
                state.pos,
                state.pos_tar,
                state.vel,
                state.vel_tar,
            )

        state_repeat = jax.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], N, axis=0), env_state
        )
        params_repeat = jax.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], N, axis=0), env_params
        )
        done_repeat = jnp.full(N, False)

        _, (pos, pos_tar, vel, vel_tar) = lax.scan(
            rollout_fn,
            (state_repeat, params_repeat, done_repeat),
            a_sampled.transpose(1, 0, 2),  # (H, N, action_dim)
            length=H,
        )
        # pos, pos_tar, vel, vel_tar: (H, N, 3)

        # state-dependent residuals (from rollout)
        r_track = (pos - pos_tar).transpose(0, 2, 1).reshape(H * 3, N)
        r_vel = (vel - vel_tar).transpose(0, 2, 1).reshape(H * 3, N)

        # action-dependent residuals (no rollout needed)
        a = a_sampled.transpose(1, 0, 2)  # (H, N, action_dim)
        r_smooth = (a[1:] - a[:-1]).transpose(0, 2, 1).reshape((H - 1) * ad, N)
        r_effort = (a - a_hover[None, None, :]).transpose(0, 2, 1).reshape(H * ad, N)

        G = jnp.concatenate(
            [
                jnp.sqrt(gw[0]) * r_track,
                jnp.sqrt(gw[1]) * r_vel,
                jnp.sqrt(gw[2]) * r_smooth,
                jnp.sqrt(gw[3]) * r_effort,
            ],
            axis=0,
        )  # (p, N)
        return G, pos

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
        # inject noise to state elements (same convention as MPPI/CoVO)
        env_state = info["noisy_state"]

        # shift operator (receding-horizon warm start)
        a_mean_old = control_params.a_mean
        a_mean = jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]])
        control_params = control_params.replace(a_mean=a_mean)

        d = self.H * self.action_dim
        gw = control_params.gw
        a_hover = control_params.a_hover

        # sample ensemble V (d, N): fixed isotropic Gaussian around the mean
        rng_act, sample_key = jax.random.split(rng_act)
        sigma_vec = jnp.tile(control_params.sample_sigma, self.H)  # (d,)
        noise = jax.random.normal(sample_key, (d, self.N))
        V = a_mean.flatten()[:, None] + sigma_vec[:, None] * noise

        # EKS transport: n_inner Gauss-Newton/Tikhonov Kalman updates
        def inner(carry, _):
            V, key = carry
            key, roll_key = jax.random.split(key)
            a_sampled = V.T.reshape(self.N, self.H, self.action_dim)
            a_sampled = jnp.clip(a_sampled, -1.0, 1.0)
            G, pos = self._rollout_residuals(
                env_state, env_params, a_sampled, gw, a_hover, roll_key
            )
            p = G.shape[0]
            Vc = V - V.mean(1, keepdims=True)
            Gc = G - G.mean(1, keepdims=True)
            Cug = (Vc @ Gc.T) / self.N
            Cgg = (Gc @ Gc.T) / self.N
            K = Cug @ jnp.linalg.inv(Cgg + (1.0 / self.alpha) * jnp.eye(p))
            V = V - K @ G
            # multiplicative inflation around the new mean
            vbar = V.mean(1, keepdims=True)
            V = vbar + (V - vbar) * self.inflate
            return (V, key), (pos.mean(1), pos.std(1))

        (V, _), (pos_mean_seq, pos_std_seq) = lax.scan(
            inner, (V, rng_act), None, length=self.n_inner
        )

        # posterior mean and covariance
        Vc = V - V.mean(1, keepdims=True)
        a_cov = (Vc @ Vc.T) / self.N
        a_mean_new = V.mean(1).reshape(self.H, self.action_dim)
        control_params = control_params.replace(a_mean=a_mean_new, a_cov=a_cov)

        # first control of the optimized sequence
        u = a_mean_new[0]

        # debug values from the final transport iteration
        info = {"pos_mean": pos_mean_seq[-1], "pos_std": pos_std_seq[-1]}

        return u, control_params, info
