"""Use sparse human critical marks without treating unmarked frames as negatives."""

import numpy as np


def phase_values(dataset):
    if "anchor_phase" not in dataset.table.column_names:
        return np.zeros(dataset.table.num_rows, dtype=np.float32)
    values = np.asarray(dataset.table["anchor_phase"], dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, [0., 1.]).all():
        raise ValueError("Expected binary human critical marks")
    return values


def training_epoch_order(indices, seed, phases=None, critical_fraction=None):
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if critical_fraction is None:
        return rng.permutation(indices)
    if not 0 < critical_fraction < 1 or phases is None:
        raise ValueError("Critical sampling requires phase marks and a fraction between zero and one")
    marked = indices[np.asarray(phases)[indices] > 0]
    unmarked = indices[np.asarray(phases)[indices] == 0]
    if not len(marked) or not len(unmarked):
        raise ValueError("Critical sampling requires both marked and unmarked training windows")
    marked_count = int(round(len(indices) * critical_fraction))
    order = np.concatenate((rng.choice(marked, marked_count, replace=True),
                            rng.choice(unmarked, len(indices) - marked_count, replace=True)))
    rng.shuffle(order)
    return order


def phase_name(value):
    return "marked_critical" if value > 0 else "unmarked"
