"""
Action-chunk buffer with optional temporal ensembling.

ACT predicts 50 actions per inference call. The buffer:
  - Caches the chunk
  - Tracks where we are in it
  - Optionally blends new chunks with old ones at chunk boundaries (smoothing)

Used by both run_local.py and run_remote.py — the chunk-handling logic is
identical regardless of where inference happens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkBufferConfig:
    chunk_execute_len: int = 30
    smoothing_enabled: bool = True
    smoothing_window: int = 8
    stale_timeout_ms: int = 500


class ChunkBuffer:
    """Holds the most recent chunk(s) and produces an action per tick."""

    def __init__(self, cfg: ChunkBufferConfig):
        self.cfg = cfg
        self.current_chunk: Optional[np.ndarray] = None
        self.current_index: int = 0
        self.current_received_ms: float = 0.0
        self.prev_chunk: Optional[np.ndarray] = None
        self.prev_index: int = 0

    def install_new_chunk(self, chunk: np.ndarray | list) -> None:
        new_chunk = np.asarray(chunk, dtype=np.float32)
        if self.current_chunk is not None:
            self.prev_chunk = self.current_chunk
            self.prev_index = self.current_index
        self.current_chunk = new_chunk
        self.current_index = 0
        self.current_received_ms = time.monotonic() * 1000

    def needs_new_chunk(self) -> bool:
        if self.current_chunk is None:
            return True
        return self.current_index >= self.cfg.chunk_execute_len

    def is_stale(self) -> bool:
        if self.current_chunk is None:
            return True
        if self.current_index < self.cfg.chunk_execute_len:
            return False
        elapsed = time.monotonic() * 1000 - self.current_received_ms
        return elapsed > self.cfg.stale_timeout_ms

    def next_action(self) -> Optional[np.ndarray]:
        if self.current_chunk is None or self.current_index >= len(self.current_chunk):
            return None

        action = self.current_chunk[self.current_index].copy()

        if self.cfg.smoothing_enabled and self.prev_chunk is not None:
            steps_since_swap = self.current_index
            if steps_since_swap < self.cfg.smoothing_window:
                prev_idx = self.prev_index + steps_since_swap
                if prev_idx < len(self.prev_chunk):
                    alpha = (steps_since_swap + 1) / self.cfg.smoothing_window
                    action = alpha * action + (1 - alpha) * self.prev_chunk[prev_idx]
            else:
                self.prev_chunk = None

        self.current_index += 1
        return action

    def reset(self) -> None:
        self.current_chunk = None
        self.prev_chunk = None
        self.current_index = 0
