import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.ppo import PPO


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str | None = None,
        verbose: int = 0,
    ):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(
                    self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl"
                )
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


@dataclass
class Settings:
    learning_rate: float = 3e-3
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: float | None = None
    net_arch_pi: list[int] = (16, 16)
    net_arch_vf: list[int] = (32, 32)


def affine_schedule(y_0: float, y_1: float) -> Callable[[float], float]:
    """!
    Affine schedule as a function over the [0, 1] interval.

    @param y_0 Function value at zero.
    @param y_1 Function value at one.
    @return Corresponding affine function.
    """
    diff = y_1 - y_0

    def schedule(x: float) -> float:
        """!
        Compute the current learning rate from remaining progress.

        @param x Progress decreasing from 1 (beginning) to 0.
        @return Corresponding learning rate>
        """
        return y_0 + x * diff

    return schedule


def train_ppo():
    now = datetime.now()
    log_dir = f"./logs/{now.strftime('%Y-%m-%d_%H:%M:%S')}"

    ppo_settings = Settings(net_arch_pi=(128, 128), net_arch_vf=(256, 256))
    nb_envs = 4

    vec_env = make_vec_env(
        lambda: TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        ),
        n_envs=nb_envs,
        vec_env_cls=SubprocVecEnv,
    )

    vec_env = VecNormalize(vec_env)

    ppo = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=affine_schedule(
            y_1=ppo_settings.learning_rate,  # progress_remaining=1.0
            y_0=ppo_settings.learning_rate / 3,  # progress_remaining=0.0
        ),
        n_steps=ppo_settings.n_steps,
        batch_size=ppo_settings.batch_size,
        n_epochs=ppo_settings.n_epochs,
        gamma=ppo_settings.gamma,
        gae_lambda=ppo_settings.gae_lambda,
        clip_range=ppo_settings.clip_range,
        clip_range_vf=ppo_settings.clip_range_vf,
        normalize_advantage=ppo_settings.normalize_advantage,
        ent_coef=ppo_settings.ent_coef,
        vf_coef=ppo_settings.vf_coef,
        max_grad_norm=ppo_settings.max_grad_norm,
        use_sde=ppo_settings.use_sde,
        sde_sample_freq=ppo_settings.sde_sample_freq,
        target_kl=ppo_settings.target_kl,
        tensorboard_log=log_dir,
        policy_kwargs={
            "net_arch": {
                "pi": ppo_settings.net_arch_pi,
                "vf": ppo_settings.net_arch_vf,
            },
        },
        verbose=1,
    ).learn(
        total_timesteps=5_000_000,
        callback=[
            CheckpointCallback(
                max(50_000 // nb_envs, 1_000),
                name_prefix="checkpoint",
                save_path=log_dir,
                verbose=1,
            ),
            SaveVecNormalizeCallback(
                max(50_000 // nb_envs, 1_000),
                name_prefix="checkpoint",
                save_path=log_dir,
                verbose=1,
            ),
        ],
    )

    ppo.save(f"{log_dir}/ppo_hiv_patient.zip")


class PPOAgent:
    def act(self, observation, use_random=False):
        observation = self.vec_normalize.normalize_obs(observation)
        action, _ = self.model.predict(observation)

        return action

    def save(self, path):
        pass

    def load(self):
        self.model = PPO.load("logs/2025-01-07_13:37:07/checkpoint_4000000_steps.zip")
        self.vec_normalize = VecNormalize.load(
            "logs/2025-01-07_13:37:07/checkpoint_4000000_steps.pkl",
            make_vec_env(lambda: env),
        )


ProjectAgent = PPOAgent

if __name__ == "__main__":
    train_ppo()
