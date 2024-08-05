import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
import mujoco
from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
from brax import envs
from brax.io import model
from omegaconf import OmegaConf
import wandb
import imageio
from rodent_env import RodentTracking
from trajectory_preprocess import process_clip_to_train
import pickle
import os

import warnings

warnings.filterwarnings("ignore")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

envs.register_environment("rodent", RodentTracking)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    qpos: jnp.ndarray


def make_train(config, env_args, reference_clip=None):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = (
        BraxGymnaxWrapper(
            config["ENV_NAME"], reference_clip=reference_clip, env_args=env_args
        ),
        None,
    )
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                qpos = env_state.env_state.env_state.env_state.pipeline_state.qpos
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info, qpos
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return (
                            total_loss,
                            (
                                value_loss,
                                loss_actor,
                                entropy,
                                ratio,
                                approx_kl,
                                clip_frac,
                            ),
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    loss_info = {
                        "total_loss": loss[0],
                        "value_loss": loss[1][0],
                        "actor_loss": loss[1][1],
                        "entropy": loss[1][2],
                        "ratio": loss[1][3],
                        "approx_kl": loss[1][4],
                        "clip_frac": loss[1][5],
                    }
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = loss_info
            metric["qposes"] = traj_batch.qpos[:, 0, :]
            metric["reward_rollout"] = traj_batch.reward[:, 0]
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info, params):
                    env_step = (
                        info["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    )
                    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)
                    model.save_params(f"{config['CHECKPOINT_DIR']}/{env_step}", params)

                    wandb.log(
                        {
                            "returns": info["returned_episode_returns"][-1, :].mean(),
                            "env_step": env_step,
                            **info["loss"],
                        }
                    )

                    scene_option = mujoco.MjvOption()

                    # Load mjx_model and mjx_data and set marker sites
                    root = mjcf.from_path(env_args["mjcf_path"])

                    rescale.rescale_subtree(
                        root,
                        0.9,
                        0.9,
                    )
                    mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr
                    mj_data = mujoco.MjData(mj_model)
                    renderer = mujoco.Renderer(mj_model, height=500, width=500)

                    mujoco.mj_kinematics(mj_model, mj_data)
                    video_path = f"{config['CHECKPOINT_DIR']}/{env_step}.mp4"
                    frames = []
                    with imageio.get_writer(video_path, fps=50) as video:
                        for i in range(info["qposes"].shape[0]):
                            mj_data.qpos = info["qposes"][i]
                            mujoco.mj_forward(mj_model, mj_data)

                            renderer.update_scene(
                                mj_data,
                                camera="close_profile",
                                scene_option=scene_option,
                            )
                            pixels = renderer.render()
                            video.append_data(pixels)
                            frames.append(pixels)

                    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

                    # Plot reward over rollout
                    data = [
                        [x, y]
                        for (x, y) in zip(
                            range(info["reward_rollout"].shape[0]),
                            info["reward_rollout"],
                        )
                    ]
                    table = wandb.Table(data=data, columns=["frame", "reward"])
                    wandb.log(
                        {
                            "eval/rollout_reward": wandb.plot.line(
                                table,
                                "frame",
                                "reward",
                                title="reward for each rollout frame",
                            )
                        }
                    )

                metric["update_steps"] = update_steps
                # Log every 10 update steps
                if metric["update_steps"] % 10 == 0:
                    jax.experimental.io_callback(
                        callback, None, metric, train_state.params
                    )
                update_steps = update_steps + 1

            runner_state = (train_state, env_state, last_obs, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    import time
    from datetime import datetime

    start_time = time.time()
    id = datetime.now()

    ppo_config = OmegaConf.to_container(
        OmegaConf.load("./configs/ppo_config.yml"), resolve=True
    )
    ppo_config["CHECKPOINT_DIR"] += f"/{id}"

    env_cfg = OmegaConf.to_container(
        OmegaConf.load("./configs/rodent_config.yml"), resolve=True
    )
    rng = jax.random.PRNGKey(42)

    env_args = env_cfg["env_args"]

    # Process rodent clip
    reference_clip = process_clip_to_train(
        env_cfg["stac_path"],
        start_step=env_cfg["clip_idx"] * env_args["clip_length"],
        clip_length=env_args["clip_length"],
        mjcf_path=env_args["mjcf_path"],
    )

    train_jit = jax.jit(make_train(ppo_config, env_args, reference_clip=reference_clip))

    run = wandb.init(
        project="purejaxrl_tracking",
        config={**ppo_config, **env_args},
        notes="purejaxrl",
        dir="/tmp",
    )

    wandb.run.name = f"tracking_{id}"
    print(f"Train config: \n {ppo_config}")
    print(f"Env config: \n {env_cfg}")
    print(f"anneal schedule: {ppo_config['ANNEAL_LR'] is True}")
    out = train_jit(rng)
    # Save train output in pickle
    with open(f"{ppo_config['CHECKPOINT_DIR']}/output.p", 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"done in {time.time() - start_time}")
    print(out)
