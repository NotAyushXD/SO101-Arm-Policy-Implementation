"""
Client-side chunk caching + temporal ensembling.

ACT predicts a 50-action chunk per call. The naive way to use it:
  1. Call server, get chunk of 50 actions.
  2. Execute action 0, then action 1, ..., then call server again.

Two improvements layered on top:

  Chunk-execute-len: don't execute all 50 actions before re-querying. Execute
  only the first N (default 30), then request a fresh chunk. The unused tail
  acts as a fallback if the network blips.

  Temporal ensembling (CHUNK_SMOOTHING): when a new chunk arrives, the
  current commanded action becomes a weighted average of the new chunk's
  prediction for "now" AND the old chunk's prediction for "now". This
  prevents the visible jerk that happens when the policy's prediction has
  moved between calls. ACT papers do this; toggling it off lets you see the
  difference.

Both behaviors are flags so you can A/B them in week 1.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkBufferConfig:
    chunk_execute_len: int = 30      # how many actions to execute per chunk
    smoothing_enabled: bool = True   # temporal ensembling on/off
    smoothing_window: int = 8        # blend over this many overlap steps
    stale_timeout_ms: int = 500      # halt arm if no new chunk in this long


class ChunkBuffer:
    """Holds the most recent chunk(s) and produces an action per tick."""

    def __init__(self, cfg: ChunkBufferConfig):
        self.cfg = cfg

        # The currently-executing chunk and where we are in it.
        self.current_chunk: Optional[np.ndarray] = None  # shape [T, A]
        self.current_index: int = 0
        self.current_received_ms: float = 0.0

        # The previous chunk, kept for smoothing only. We blend the first
        # `smoothing_window` actions of the new chunk with the last
        # corresponding actions of the previous chunk.
        self.prev_chunk: Optional[np.ndarray] = None
        self.prev_index: int = 0  # where prev was when we swapped

    def install_new_chunk(self, chunk: list[list[float]]) -> None:
        """Called when a new chunk arrives from the server."""
        new_chunk = np.asarray(chunk, dtype=np.float32)
        if self.current_chunk is not None:
            self.prev_chunk = self.current_chunk
            self.prev_index = self.current_index
        self.current_chunk = new_chunk
        self.current_index = 0
        self.current_received_ms = time.monotonic() * 1000

    def needs_new_chunk(self) -> bool:
        """True if we should request a new chunk now."""
        if self.current_chunk is None:
            return True
        return self.current_index >= self.cfg.chunk_execute_len

    def is_stale(self) -> bool:
        """True if the current chunk has timed out (network failed)."""
        if self.current_chunk is None:
            return True
        # Only stale if we've actually run past the executable region.
        if self.current_index < self.cfg.chunk_execute_len:
            return False
        elapsed_ms = time.monotonic() * 1000 - self.current_received_ms
        return elapsed_ms > self.cfg.stale_timeout_ms

    def next_action(self) -> Optional[np.ndarray]:
        """
        Return the action to command this tick, or None if we have nothing
        valid to send (caller should halt the arm).

        Applies temporal ensembling when smoothing_enabled.
        """
        if self.current_chunk is None:
            return None
        if self.current_index >= len(self.current_chunk):
            # Ran past the end of the chunk and no new one arrived.
            return None

        action = self.current_chunk[self.current_index].copy()

        # Temporal ensembling: blend with the previous chunk's prediction for
        # the same wall-clock step, if we're still within the smoothing window
        # of the chunk swap.
        if self.cfg.smoothing_enabled and self.prev_chunk is not None:
            steps_since_swap = self.current_index
            if steps_since_swap < self.cfg.smoothing_window:
                # Where would the previous chunk have been at this tick?
                prev_idx = self.prev_index + steps_since_swap
                if prev_idx < len(self.prev_chunk):
                    # Linear blend: 100% old → 100% new across the window.
                    alpha = (steps_since_swap + 1) / self.cfg.smoothing_window
                    action = (alpha * action +
                              (1 - alpha) * self.prev_chunk[prev_idx])
            else:
                # Smoothing window passed; can drop the prev chunk to free RAM.
                self.prev_chunk = None

        self.current_index += 1
        return action

    def reset(self) -> None:
        self.current_chunk = None
        self.prev_chunk = None
        self.current_index = 0
