"""
rl/policy.py
Lightweight Actor-Critic policy network for indoor navigation RL.

Architecture:
  Input state vector → shared MLP encoder → split into:
    • Actor head  → logits over N_ACTIONS discrete actions
    • Critic head → scalar state value V(s)

Designed to run inference in < 5 ms on the RPi 5 CPU.
All heavy vision compute is done upstream (VisionPipeline); this
network only processes the assembled flat state vector.

Optional GRU layer for temporal memory (enabled via cfg).
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np


# ── Pure-numpy fallback (no torch) ───────────────────────────────
# Used when torch is not installed (simulation / unit tests only).


class _LinearNP:
    """Single linear layer: y = x @ W.T + b (numpy)."""
    def __init__(self, in_dim: int, out_dim: int, key: int = 0):
        rng = np.random.default_rng(key)
        std = math.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * std
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ActorCriticNumpy:
    """NumPy-only Actor-Critic (inference only, no gradient)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256) -> None:
        self.shared_1 = _LinearNP(state_dim, hidden, 0)
        self.shared_2 = _LinearNP(hidden,    hidden, 1)
        self.actor    = _LinearNP(hidden, n_actions, 2)
        self.critic   = _LinearNP(hidden, 1,         3)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        x = _relu(self.shared_1(state))
        x = _relu(self.shared_2(x))
        logits = self.actor(x)
        value  = float(self.critic(x)[0])
        return logits, value

    def act(self, state: np.ndarray, deterministic: bool = False
            ) -> Tuple[int, float, float]:
        logits, value = self.forward(state)
        probs = _softmax(logits)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    def save(self, path: str) -> None:
        np.savez(path,
                 s1w=self.shared_1.W, s1b=self.shared_1.b,
                 s2w=self.shared_2.W, s2b=self.shared_2.b,
                 aw=self.actor.W,     ab=self.actor.b,
                 cw=self.critic.W,    cb=self.critic.b)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.shared_1.W = data["s1w"]; self.shared_1.b = data["s1b"]
        self.shared_2.W = data["s2w"]; self.shared_2.b = data["s2b"]
        self.actor.W    = data["aw"];  self.actor.b    = data["ab"]
        self.critic.W   = data["cw"];  self.critic.b   = data["cb"]


# ── PyTorch implementation (preferred) ───────────────────────────

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        """
        Shared-trunk Actor-Critic with optional GRU temporal layer.

        Parameters
        ----------
        state_dim  : dimension of flat state vector
        n_actions  : number of discrete actions
        hidden_dim : MLP hidden units
        use_gru    : enable single-layer GRU for temporal context
        """

        def __init__(
            self,
            state_dim:  int,
            n_actions:  int,
            hidden_dim: int = 256,
            use_gru:    bool = False,
            device:     str | None = None,
        ) -> None:
            super().__init__()
            self.use_gru = use_gru
            self.device = torch.device(
                device if device
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            _logger.info("[Policy] Using device: %s", self.device)

            # Shared encoder
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            if use_gru:
                self.gru       = nn.GRU(hidden_dim, hidden_dim,
                                         batch_first=True)
                self._gru_hidden: torch.Tensor | None = None

            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_actions),
            )

            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            self._init_weights()
            self.to(self.device)

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            # Smaller init for actor output
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        def forward(
            self,
            state: torch.Tensor,
            hidden: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """
            Returns (logits, value, hidden_state).
            hidden_state is None if use_gru=False.
            """
            x = self.shared(state)
            if self.use_gru:
                x, hidden = self.gru(x.unsqueeze(1), hidden)
                x = x.squeeze(1)
            logits = self.actor(x)
            value  = self.critic(x)
            return logits, value, hidden

        def act(
            self,
            state: np.ndarray,
            deterministic: bool = False,
            hidden=None,
        ) -> Tuple[int, float, float, object]:
            """
            Sample or argmax an action from the policy.

            Returns (action, log_prob, value, new_hidden).
            """
            with torch.no_grad():
                t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                logits, val, new_h = self.forward(t, hidden)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
            return (int(action.item()),
                    float(log_prob.item()),
                    float(val.squeeze().item()),
                    new_h)

        def evaluate(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate batch of (state, action) pairs for PPO update.

            Returns (log_probs, values, entropy).
            """
            logits, values, _ = self.forward(states)
            dist     = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy  = dist.entropy()
            return log_prob, values.squeeze(-1), entropy

        def reset_hidden(self) -> None:
            self._gru_hidden = None

except ImportError:
    # torch not available – alias numpy version
    ActorCritic = ActorCriticNumpy  # type: ignore[misc]
