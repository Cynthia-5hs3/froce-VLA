"""Timestamped torque conditioning and latency-aware action chunk consumption."""

from collections import deque

import numpy as np


SAMPLE_RATE_HZ = 30.0
HISTORY_OFFSETS = (-60, -53, -47, -40, -33, -27, -20, -13, -7, 0)  
# 2-second history抽取10帧


class HistoryNotReady(RuntimeError):
    pass


class TorqueHistory:
    def __init__(self, fps=SAMPLE_RATE_HZ, offsets=HISTORY_OFFSETS):
        self.offsets_ns = np.rint(np.asarray(offsets) * 1e9 / fps).astype(np.int64)
        if len(offsets) != 10 or offsets[-1] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("Expected ten increasing causal offsets ending at zero")
        self.tolerance_ns = int((1 / fps - 1e-4) * 1e9)
        self.samples = deque()

    def clear(self):
        self.samples.clear()

    def append(self, timestamp_ns, torque):
        value = np.asarray(torque, dtype=np.float32)
        if timestamp_ns <= 0 or value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("Invalid timestamped joint torque")
        if self.samples and timestamp_ns <= self.samples[-1][0]:
            raise ValueError("Torque timestamps must increase")
        self.samples.append((int(timestamp_ns), value.copy()))
        cutoff = timestamp_ns + self.offsets_ns[0] - int(1e9)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def sample(self, anchor_ns):
        causal = [sample for sample in self.samples if sample[0] <= anchor_ns]
        queries = int(anchor_ns) + self.offsets_ns
        if not causal or causal[0][0] > queries[0] or causal[-1][0] < anchor_ns:
            raise HistoryNotReady("Waiting for a complete two-second torque history")
        timestamps = np.asarray([sample[0] for sample in causal], dtype=np.int64)
        positions = np.clip(np.searchsorted(timestamps, queries), 0, len(timestamps) - 1)
        previous = np.maximum(positions - 1, 0)
        positions = np.where(np.abs(timestamps[previous] - queries) <= np.abs(timestamps[positions] - queries),
                             previous, positions)
        selected = timestamps[positions]
        if (np.any(np.abs(selected - queries) > self.tolerance_ns)
                or np.any(np.diff(positions) <= 0)
                or np.any(np.diff(timestamps[positions[0]:]) > 100_000_000)):
            raise HistoryNotReady("Torque history has missing or irregular samples")
        values = np.stack([causal[int(position)][1] for position in positions])
        return values, selected.copy()


def blend_pose(old, new, fraction):
    result = np.asarray(new, dtype=np.float32).copy()
    result[:9] = (1 - fraction) * np.asarray(old)[:9] + fraction * result[:9]
    first = result[3:6]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-6:
        raise ValueError("Cannot blend degenerate rotations")
    first = first / first_norm
    second = result[6:9] - np.dot(first, result[6:9]) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-6:
        raise ValueError("Cannot blend degenerate rotations")
    result[3:6], result[6:9] = first, second / second_norm
    return result


class ActionChunks:
    def __init__(self, fps=SAMPLE_RATE_HZ, blend_steps=3):
        if fps <= 0 or blend_steps < 0:
            raise ValueError("Invalid chunk execution settings")
        self.fps = fps
        self.blend_steps = blend_steps
        self.chunk = None
        self.sequence = 0
        self.valid_after_ns = 0

    def invalidate(self, timestamp_ns):
        self.chunk = None
        self.valid_after_ns = int(timestamp_ns)

    def _index(self, chunk, timestamp_ns):
        elapsed = (int(timestamp_ns) - int(chunk["anchor_ns"])) / 1e9
        return int(np.floor(elapsed * self.fps + 1e-7))

    def _sample(self, chunk, timestamp_ns):
        if chunk is None:
            return None
        index = self._index(chunk, timestamp_ns)
        if index < 0 or index >= len(chunk["actions"]):
            return None
        return chunk["actions"][index].copy(), chunk["torques"][index].copy(), index

    def install(self, result, now_ns, fallback_action):
        if result is None or result["sequence"] <= self.sequence:
            return False
        self.sequence = int(result["sequence"])
        if result["anchor_ns"] < self.valid_after_ns or result["anchor_ns"] > now_ns:
            return False
        actions = np.asarray(result["actions"], dtype=np.float32).copy()
        torques = np.asarray(result["torques"], dtype=np.float32).copy()
        if actions.shape != (50, 10) or torques.shape != (50, 7):
            raise ValueError("Expected a 50-step action/torque chunk")
        if not np.isfinite(actions).all() or not np.isfinite(torques).all():
            raise ValueError("Nonfinite action chunk")
        candidate = dict(result, actions=actions, torques=torques, raw_actions=actions.copy())
        first = self._index(candidate, now_ns)
        if first >= len(actions):
            return False
        for offset in range(min(self.blend_steps, len(actions) - first)):
            index = first + offset
            at_ns = result["anchor_ns"] + int((index + 0.5) / self.fps * 1e9)
            old = self._sample(self.chunk, at_ns)
            origin = fallback_action if old is None else old[0]
            actions[index] = blend_pose(origin, actions[index], (offset + 1) / (self.blend_steps + 1))
        candidate["skipped_steps_on_arrival"] = first
        self.chunk = candidate
        return True

    def sample(self, now_ns):
        sampled = self._sample(self.chunk, now_ns)
        if sampled is None:
            return None
        action, torque, index = sampled
        return {"action": action, "predicted_torque": torque, "action_index": index,
                "model_action": self.chunk["raw_actions"][index].copy(),
                "sequence": self.chunk["sequence"], "anchor_ns": self.chunk["anchor_ns"],
                "inference_ms": self.chunk["inference_ms"],
                "history_timestamps_ns": self.chunk["history_timestamps_ns"],
                "skipped_steps_on_arrival": self.chunk["skipped_steps_on_arrival"]}
