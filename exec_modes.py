"""
Execution mode dispatch. Each mode wraps the "send action to robot" call with
different safety behavior.

  shadow:     don't move at all; just log what would have happened.
  visualize:  don't move; render predicted trajectory in the web UI.
  live_slow:  move, but scale velocity by a factor (default 0.3).
  live:       move at full speed.

The progression is deliberate: shadow → visualize → live_slow → live. Don't
skip steps on the first session of a new policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class RobotLike(Protocol):
    """Minimal interface so we can mock it in tests."""
    def get_observation(self) -> dict: ...
    def send_action(self, action: np.ndarray) -> None: ...


@dataclass
class ExecModeConfig:
    mode: str = "shadow"
    live_slow_factor: float = 0.3


class ExecModeRunner:
    """
    Wraps robot.send_action with mode-aware behavior.

    Visualization data (predicted vs commanded) is stored for the caller to
    push to the UI / log to W&B.
    """
    def __init__(self, cfg: ExecModeConfig):
        self.cfg = cfg
        self.last_predicted: Optional[np.ndarray] = None
        self.last_commanded: Optional[np.ndarray] = None
        self.last_joint_state: Optional[np.ndarray] = None

    @property
    def will_move(self) -> bool:
        return self.cfg.mode in {"live_slow", "live"}

    def step(self, robot: RobotLike, predicted_action: np.ndarray,
             current_state: np.ndarray) -> np.ndarray:
        """
        Apply the policy's predicted action under the current mode.

        Returns the action that was actually commanded (may differ from
        predicted under live_slow scaling). The robot is only mutated in
        live / live_slow modes.
        """
        self.last_predicted = predicted_action.copy()
        self.last_joint_state = current_state.copy()

        if self.cfg.mode in {"shadow", "visualize"}:
            # Don't touch the robot. Return predicted so the caller can log.
            self.last_commanded = predicted_action.copy()
            return predicted_action

        if self.cfg.mode == "live_slow":
            # Scale the *delta* from current state, not the absolute target.
            # This makes the arm move toward the predicted target, just slower.
            commanded = current_state + self.cfg.live_slow_factor * (
                predicted_action - current_state
            )
        elif self.cfg.mode == "live":
            commanded = predicted_action
        else:
            raise ValueError(f"Unknown exec mode: {self.cfg.mode}")

        robot.send_action(commanded.astype(np.float32))
        self.last_commanded = commanded.copy()
        return commanded

    def halt(self, robot: RobotLike) -> None:
        """Called when chunk goes stale or trial is aborted."""
        if not self.will_move:
            return
        if self.last_joint_state is None:
            return
        # Hold position by re-commanding the current state.
        robot.send_action(self.last_joint_state.astype(np.float32))
