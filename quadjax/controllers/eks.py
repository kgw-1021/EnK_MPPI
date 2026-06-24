import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp

from quadjax import controllers


@struct.dataclass
class EKSParams:
    gamma_mean: float  # mean of gamma
    gamma_sigma: float  # std of gamma
    discount: float  # discount factor
    sample_sigma: float  # std of sampling

    a_mean: jnp.ndarray  # mean of action
    a_cov: jnp.ndarray  # covariance matrix of action


class EKSController(controllers.BaseController):
    """Ensemble Kalman Sampler controller.

    Identical to MPPI in parameters, isotropic sampling, rollout, and the
    (reward-based) cost -- the only difference is the distribution update:
    MPPI's softmax weighting + weighted mean is replaced by the EKS transport.

    The same cost is used (no hand-designed track/smooth/effort residuals); the
    only change is that the per-step rewards are kept as an H-dimensional
    observation G = -reward_seq (H, N) instead of being collapsed to a single
    scalar, so the ensemble Kalman update has multiple directions to transport
    along. Each inner iteration re-rolls out the transported ensemble and applies

        K = C^UG (C^GG + (1/alpha) I)^-1,   V <- V - K (G - y),   then inflate,

    with y the per-step best (lowest) cost across the ensemble. The posterior
    ensemble mean's first control is returned.
    """

    def __init__(
        self,
        env,
        control_params,
        N: int,
        H: int,
        lam: float,
        n_inner: int = 5,
        alpha: float = 10.0,
        inflate: float = 1.0,
    ) -> None:
        super().__init__(env, control_params)
        self.N = N  # NOTE: N is the number of samples, set here as a static number
        self.H = H
        self.lam = lam  # kept for interface parity with MPPI/CoVO (unused by EKS)
        self.n_inner = n_inner  # EKS transport iterations
        self.alpha = alpha  # Tikhonov regularization (1/alpha on the C^GG diagonal)
        self.inflate = inflate  # ensemble inflation factor per inner iteration
        self.action_dim = env.action_dim

    def _rollout_rewards(self, env_state, env_params, a_sampled, step_key):
        """Roll out N action sequences; return per-step rewards (H, N) and poses."""

        def rollout_fn(carry, action):
            env_state, params, reward_before, done_before = carry
            obs, env_state, reward, done, info = jax.vmap(
                lambda s, a, p: self.env.step_env(step_key, s, a, p, True)
            )(env_state, action, params)
            reward = jnp.where(done_before, reward_before, reward)
            return (env_state, params, reward, done | done_before), (
                reward,
                env_state.pos,
            )

        state_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_state
        )
        env_params_repeat = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_params
        )
        done_repeat = jnp.full(self.N, False)
        reward_repeat = jnp.full(self.N, 0.0)

        _, (rewards, poses) = lax.scan(
            rollout_fn,
            (state_repeat, env_params_repeat, reward_repeat, done_repeat),
            a_sampled.transpose(1, 0, 2),
            length=self.H,
        )
        return rewards, poses  # (H, N), (H, N, 3)

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

        # per-step discount weights (sqrt, so squared-misfit matches discounted cost)
        disc = jnp.sqrt(jnp.power(control_params.discount, jnp.arange(self.H)))  # (H,)

        # ensemble of flattened action samples: V (d, N), d = H * action_dim
        V = a_sampled.reshape(self.N, -1).T

        # === EKS transport (replaces MPPI softmax weighting + weighted mean) ===
        def inner(carry, _):
            V, key = carry
            key, step_key = jax.random.split(key)
            a_cur = jnp.clip(V.T.reshape(self.N, self.H, self.action_dim), -1.0, 1.0)
            rewards, poses = self._rollout_rewards(
                env_state, env_params, a_cur, step_key
            )
            # observation: per-step cost (drive down), discounted; (H, N)
            G = (-rewards) * disc[:, None]
            y = jnp.min(G, axis=1, keepdims=True)  # per-step best member (H, 1)

            Vc = V - V.mean(axis=1, keepdims=True)
            Gc = G - G.mean(axis=1, keepdims=True)
            Cug = (Vc @ Gc.T) / self.N  # (d, H)
            Cgg = (Gc @ Gc.T) / self.N  # (H, H)
            # relative Tikhonov regularization (scale-invariant in the cost units):
            # larger alpha -> smaller reg -> closer to the full Gauss-Newton step
            reg = jnp.trace(Cgg) / self.H / self.alpha + 1e-8
            K = Cug @ jnp.linalg.inv(Cgg + reg * jnp.eye(self.H))
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
