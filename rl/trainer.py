"""
rl/trainer.py
Training loop for the PPO navigation agent.

Supports two training modes:

  "sim"   – run entirely in the built-in simulation environment (no
             hardware needed).  Fast; use on a laptop / desktop first.

  "robot" – train on the real robot.  Same code, slower dynamics,
             real sensor noise.  Robot safe-stop is enforced when
             any sensor reads < emergency_stop_cm.

Usage:
    trainer = Trainer(cfg, env, agent)
    trainer.train(n_episodes=500)
    trainer.evaluate(n_episodes=10)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """
    Parameters
    ----------
    cfg   : full config dict
    env   : RobotEnv instance
    agent : PPOAgent instance
    """

    def __init__(self, cfg: dict, env, agent) -> None:
        self._cfg   = cfg
        self._env   = env
        self._agent = agent
        self._log_dir = cfg.get("agent", {}).get("logging", {}).get(
                            "log_dir", "logs/")
        os.makedirs(self._log_dir, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────

    def train(self, n_episodes: int = 500) -> None:
        """Run PPO training for ``n_episodes`` episodes."""
        logger.info("[Trainer] Starting training  n_episodes=%d", n_episodes)
        t_start = time.monotonic()
        global_step = 0

        for episode in range(1, n_episodes + 1):
            state   = self._env.reset()
            ep_reward = 0.0
            ep_length = 0
            done  = False

            while not done:
                action, log_prob, value = self._agent.select_action(state)
                next_state, reward, done, info = self._env.step(action)

                self._agent.store(state, action, log_prob, reward, done, value)
                ep_reward += reward
                ep_length += 1
                global_step += 1
                state = next_state

                # Rollout update
                if self._agent.ready_to_update():
                    # Bootstrap value for last state
                    _, _, last_val = self._agent.select_action(state)
                    stats = self._agent.update(last_val if not done else 0.0)
                    self._log_stats(stats, global_step)

            self._agent.on_episode_end(ep_reward, ep_length)

            if episode % 10 == 0:
                elapsed = time.monotonic() - t_start
                logger.info("[Trainer] ep=%d/%d  step=%d  %.0f s elapsed",
                             episode, n_episodes, global_step, elapsed)

        logger.info("[Trainer] Training complete (%d episodes)", n_episodes)

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Run greedy evaluation.

        Returns
        -------
        dict with mean reward, mean length, mean explored cells.
        """
        logger.info("[Trainer] Evaluating  n=%d episodes", n_episodes)
        rewards  = []
        lengths  = []
        explored = []

        for ep in range(n_episodes):
            state = self._env.reset()
            ep_r  = 0.0
            ep_l  = 0
            done  = False
            while not done:
                action, _, _ = self._agent.select_action(state,
                                                          deterministic=True)
                state, reward, done, info = self._env.step(action)
                ep_r += reward
                ep_l += 1
            rewards.append(ep_r)
            lengths.append(ep_l)
            explored.append(self._env._grid.explored_cells
                            if self._env._grid else 0)

        results = {
            "mean_reward":   float(np.mean(rewards)),
            "std_reward":    float(np.std(rewards)),
            "mean_length":   float(np.mean(lengths)),
            "mean_explored": float(np.mean(explored)),
        }
        logger.info("[Trainer] Eval results: %s", results)
        return results

    # ── Logging ───────────────────────────────────────────────────

    def _log_stats(self, stats: dict, step: int) -> None:
        if not stats:
            return
        logger.debug("[Trainer] step=%d  %s", step,
                     "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        # Optionally write CSV
        csv_path = os.path.join(self._log_dir, "train_stats.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if header:
                f.write("step," + ",".join(stats.keys()) + "\n")
            f.write(str(step) + "," +
                    ",".join(f"{v:.6f}" for v in stats.values()) + "\n")
