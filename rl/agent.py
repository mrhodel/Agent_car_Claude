"""
rl/agent.py
Proximal Policy Optimisation (PPO-clip) agent.

Implements the full PPO update from "Proximal Policy Optimization
Algorithms" (Schulman et al., 2017).

Key features:
  • Generalised Advantage Estimation (GAE).
  • Clipped surrogate objective.
  • Combined actor + critic + entropy loss.
  • Checkpoint save / load.
  • Graceful degradation if torch is not available (inference only).
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Stores one rollout of (s, a, log_p, reward, done, value)."""

    def __init__(self) -> None:
        self.states:    List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.dones:     List[bool]       = []
        self.values:    List[float]      = []

    def push(self, state, action, log_prob, reward, done, value) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    """
    Proximal Policy Optimisation agent.

    Parameters
    ----------
    policy      : ActorCritic network
    cfg         : the ``rl.ppo`` sub-dict from robot_config.yaml
    state_dim   : input dimension
    n_actions   : number of discrete actions
    """

    def __init__(self, policy, cfg: dict, state_dim: int,
                 n_actions: int) -> None:
        self._policy    = policy
        self._state_dim = state_dim
        self._n_actions = n_actions

        self._lr          = float(cfg.get("learning_rate",      3e-4))
        self._gamma       = float(cfg.get("gamma",              0.99))
        self._gae_lambda  = float(cfg.get("gae_lambda",         0.95))
        self._clip_eps    = float(cfg.get("clip_epsilon",        0.20))
        self._vf_coeff    = float(cfg.get("value_loss_coeff",   0.50))
        self._ent_coeff   = float(cfg.get("entropy_coeff",      0.01))
        self._epochs      = int(cfg.get("epochs_per_update",     4))
        self._batch_size  = int(cfg.get("batch_size",           64))
        self._rollout_len = int(cfg.get("rollout_steps",        512))
        self._ckpt_dir    = cfg.get("checkpoint_dir", "models/checkpoints")
        self._ckpt_iv     = int(cfg.get("checkpoint_interval",  50))

        os.makedirs(self._ckpt_dir, exist_ok=True)

        self._buffer  = RolloutBuffer()
        self._episode = 0
        self._opt     = None
        self._torch_available = False

        self._init_optimizer()

        # Running reward stats
        self._ep_rewards: deque = deque(maxlen=100)
        self._ep_lengths: deque = deque(maxlen=100)

    # ── Initialisation ────────────────────────────────────────────

    def _init_optimizer(self) -> None:
        try:
            import torch.optim as optim
            self._opt = optim.Adam(self._policy.parameters(), lr=self._lr,
                                    eps=1e-5)
            self._torch_available = True
            logger.info("[PPO] Adam optimiser  lr=%.1e", self._lr)
        except (ImportError, AttributeError):
            logger.warning("[PPO] torch not available – inference only mode")

    # ── Interaction API ───────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select an action from the current policy.

        Returns (action, log_prob, value).
        """
        if self._torch_available:
            action, lp, val, _ = self._policy.act(state, deterministic)
        else:
            action, lp, val = self._policy.act(state, deterministic)
        return action, lp, val

    def store(self, state, action, log_prob, reward, done, value) -> None:
        self._buffer.push(state, action, log_prob, reward, done, value)

    def ready_to_update(self) -> bool:
        return len(self._buffer) >= self._rollout_len

    def update(self, last_value: float = 0.0) -> dict:
        """
        Run PPO update on the collected rollout.

        Parameters
        ----------
        last_value : bootstrap value for the last state (0 if terminal).

        Returns
        -------
        dict with loss statistics.
        """
        if not self._torch_available:
            self._buffer.clear()
            return {}

        import torch

        buf = self._buffer

        # -- Compute GAE advantages --
        advantages, returns = self._compute_gae(
            buf.rewards, buf.values, buf.dones, last_value
        )

        # Convert to tensors
        dev = self._policy.device if hasattr(self._policy, 'device') else torch.device('cpu')
        states    = torch.FloatTensor(np.array(buf.states)).to(dev)
        actions   = torch.LongTensor(buf.actions).to(dev)
        old_lps   = torch.FloatTensor(buf.log_probs).to(dev)
        advs      = torch.FloatTensor(advantages).to(dev)
        rets      = torch.FloatTensor(returns).to(dev)
        advs      = (advs - advs.mean()) / (advs.std() + 1e-8)  # normalise

        self._buffer.clear()

        total_ploss = total_vloss = total_eloss = 0.0
        n_updates = 0

        for _ in range(self._epochs):
            # Random mini-batches
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), self._batch_size):
                batch = idx[start:start + self._batch_size]
                s_b  = states[batch];  a_b  = actions[batch]
                op_b = old_lps[batch]; ad_b = advs[batch]
                re_b = rets[batch]

                new_lp, values, entropy = self._policy.evaluate(s_b, a_b)

                # Policy loss (PPO-clip)
                ratio  = torch.exp(new_lp - op_b)
                obj1   = ratio * ad_b
                obj2   = torch.clamp(ratio, 1 - self._clip_eps,
                                      1 + self._clip_eps) * ad_b
                p_loss = -torch.min(obj1, obj2).mean()

                # Value loss (clipped MSE)
                v_loss = 0.5 * (values - re_b).pow(2).mean()

                # Entropy bonus
                e_loss = -entropy.mean()

                loss = (p_loss
                        + self._vf_coeff * v_loss
                        + self._ent_coeff * e_loss)

                self._opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), max_norm=0.5)
                self._opt.step()

                total_ploss += p_loss.item()
                total_vloss += v_loss.item()
                total_eloss += e_loss.item()
                n_updates   += 1

        n = max(1, n_updates)
        stats = {
            "p_loss": total_ploss / n,
            "v_loss": total_vloss / n,
            "e_loss": total_eloss / n,
        }
        logger.debug("[PPO] update  p=%.4f  v=%.4f  e=%.4f",
                     stats["p_loss"], stats["v_loss"], stats["e_loss"])
        return stats

    def on_episode_end(self, ep_reward: float, ep_length: int) -> None:
        self._episode += 1
        self._ep_rewards.append(ep_reward)
        self._ep_lengths.append(ep_length)

        mean_r = np.mean(self._ep_rewards)
        logger.info("[PPO] ep=%d  R=%.1f  mean100=%.1f  len=%d",
                    self._episode, ep_reward, mean_r, ep_length)

        if self._episode % self._ckpt_iv == 0:
            self.save_checkpoint(f"ep{self._episode}")

    def save_checkpoint(self, tag: str = "") -> None:
        if self._torch_available:
            import torch
            path = os.path.join(self._ckpt_dir, f"policy_{tag}.pt")
            torch.save(self._policy.state_dict(), path)
            logger.info("[PPO] Checkpoint saved -> %s", path)
        else:
            path = os.path.join(self._ckpt_dir, f"policy_{tag}.npz")
            self._policy.save(path)

    def load_checkpoint(self, path: str) -> None:
        if self._torch_available:
            import torch
            self._policy.load_state_dict(
                torch.load(path, map_location="cpu"))
            logger.info("[PPO] Checkpoint loaded <- %s", path)
        else:
            self._policy.load(path)

    # ── GAE ───────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards:    List[float],
        values:     List[float],
        dones:      List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_val = last_value
        for t in reversed(range(n)):
            mask   = 0.0 if dones[t] else 1.0
            delta  = (rewards[t]
                      + self._gamma * next_val * mask
                      - values[t])
            gae    = delta + self._gamma * self._gae_lambda * mask * gae
            advantages[t] = gae
            next_val = values[t]
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns
