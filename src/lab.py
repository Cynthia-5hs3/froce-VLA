from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [Path('/data0/sht/Evo-RLT'), Path('/data0/miniconda3/envs/evo-rlt')]


def isolation_report():
    errors = []
    if not Path(sys.prefix).resolve().is_relative_to(ROOT / 'env'):
        errors.append(f'Unexpected Python prefix: {sys.prefix}')
    for entry in sys.path:
        if any(Path(entry).resolve().is_relative_to(path) for path in FORBIDDEN) or '/opt/ros' in entry:
            errors.append(f'Foreign import path: {entry}')
    mounts = {}
    for path in [*FORBIDDEN, ROOT / 'datasets', ROOT / 'models', ROOT / 'reference']:
        mounts[str(path)] = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
        if not mounts[str(path)]:
            errors.append(f'Expected read-only mount: {path}')
    imports = {}
    for name in ['torch', 'lerobot', 'evo_rlt', 'pyarrow']:
        specification = importlib.util.find_spec(name)
        imports[name] = None if specification is None else specification.origin
        if specification is None or not Path(specification.origin).resolve().is_relative_to(ROOT):
            errors.append(f'Non-local dependency: {name}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    return {'python': sys.executable, 'prefix': sys.prefix, 'imports': imports,
            'readonly_mounts': mounts, 'home': os.environ['HOME'], 'errors': errors}


def load_episode(path):
    table = pq.read_table(path)
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    observed = np.asarray(table['complementary_info.observed_at_ns'], dtype=np.int64)
    timestamps = (observed - observed[0]).astype(np.float64) / 1e9
    valid = np.asarray(table['complementary_info.action_valid']).astype(bool) if 'complementary_info.action_valid' in table.column_names else np.ones(len(state), dtype=bool)
    segments = np.asarray(table['complementary_info.segment_id']) if 'complementary_info.segment_id' in table.column_names else np.zeros(len(state), dtype=np.int64)
    return {'state': state, 'action': action, 'time': timestamps, 'valid': valid, 'segment': segments}


def make_window(episode, anchor, history_seconds=2.0, history_frames=10, future_steps=50):
    timestamps = episode['time']
    if anchor < 0 or anchor + future_steps > len(timestamps):
        return None
    desired = np.linspace(timestamps[anchor] - history_seconds, timestamps[anchor], history_frames)
    if desired[0] < timestamps[0]:
        return None
    indices = np.searchsorted(timestamps, desired, side='right') - 1
    start = int(indices[0])
    end = anchor + future_steps
    if not np.all(np.diff(timestamps[start:end]) > 0) or np.any(np.diff(timestamps[start:end]) > 0.1):
        return None
    if not np.all(episode['segment'][start:end] == episode['segment'][anchor]):
        return None
    if not episode['valid'][anchor:end].all():
        return None
    history = episode['state'][indices, 10:16]
    future = episode['state'][anchor:end, 10:16]
    actions = episode['action'][anchor:end]
    if not all(np.isfinite(values).all() for values in [history, future, actions]):
        return None
    return {'history': history.copy(), 'future_wrench': future.copy(),
            'future_actions': actions.copy(), 'history_indices': indices.copy()}


def check_environment():
    result = {'isolation': isolation_report()}
    import importlib.metadata
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/tokenizer'), local_files_only=True)
    tokens = tokenizer('Place the white lid firmly onto the jig.', return_tensors='pt')
    with safe_open(str(ROOT / 'models/pi05_sft/model.safetensors'), framework='pt', device='cpu') as weights:
        shapes = {key: weights.get_slice(key).get_shape() for key in weights.keys()
                  if key.endswith(('action_in_proj.weight', 'action_out_proj.weight'))}
        count = len(weights.keys())
    result.update({'versions': {name: importlib.metadata.version(name)
                               for name in ['torch', 'lerobot', 'transformers', 'pyarrow', 'safetensors']},
                   'policy_class_imported': PI05Policy.__name__, 'tokenizer_class': type(tokenizer).__name__,
                   'example_token_shape': list(tokens['input_ids'].shape),
                   'checkpoint_tensor_count': count, 'checkpoint_action_projection_shapes': shapes,
                   'full_checkpoint_loaded': False, 'cuda_available': torch.cuda.is_available(),
                   'cuda_build': torch.version.cuda})
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
        result['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    (ROOT / 'outputs/environment_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def inspect():
    result = {'isolation': isolation_report(), 'datasets': {}}
    for root in sorted((ROOT / 'datasets/raw').iterdir()):
        info = json.loads((root / 'meta/info.json').read_text())
        tasks = pq.read_table(root / 'meta/tasks.parquet').to_pylist()
        files = sorted(root.glob('data/**/*.parquet'))
        frames = 0
        nonfinite = 0
        invalid_actions = 0
        force_min = np.full(6, np.inf)
        force_max = np.full(6, -np.inf)
        for path in files:
            episode = load_episode(path)
            frames += len(episode['state'])
            nonfinite += int((~np.isfinite(episode['state'])).sum() + (~np.isfinite(episode['action'])).sum())
            invalid_actions += int((~episode['valid']).sum())
            force_min = np.minimum(force_min, episode['state'][:, 10:16].min(axis=0))
            force_max = np.maximum(force_max, episode['state'][:, 10:16].max(axis=0))
        episode_files = sorted(root.glob('meta/episodes/**/*.parquet'))
        outcomes = Counter()
        for path in episode_files:
            table = pq.read_table(path)
            if 'episode_success' in table.column_names:
                outcomes.update(table['episode_success'].to_pylist())
        if frames != info['total_frames'] or len(files) != info['total_episodes']:
            raise RuntimeError(f'Dataset count mismatch: {root}')
        result['datasets'][root.name] = {'episodes': info['total_episodes'], 'frames': frames,
            'fps': info['fps'], 'tasks': tasks, 'outcomes': dict(outcomes), 'nonfinite': nonfinite,
            'invalid_actions': invalid_actions, 'wrench_min': force_min.tolist(), 'wrench_max': force_max.tolist(),
            'placeholder_task': any(task['task'].strip() == '111' for task in tasks)}
    output = ROOT / 'outputs/data_readiness.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


def smoke(steps, device):
    isolation_report()
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.manual_seed(42)
    files = sorted((ROOT / 'datasets/raw/gamepad_task3_20260909_online_c20_full').glob('data/**/*.parquet'))
    train_windows = []
    validation_windows = []
    for paths, windows in [(files[:4], train_windows), (files[-2:], validation_windows)]:
        for path in paths:
            episode = load_episode(path)
            for anchor in range(0, len(episode['time']), 10):
                window = make_window(episode, anchor)
                if window is not None:
                    windows.append(window)
    if not train_windows or not validation_windows:
        raise RuntimeError('No valid trajectory windows')
    train_history = torch.from_numpy(np.stack([window['history'] for window in train_windows]))
    train_future = torch.from_numpy(np.stack([window['future_wrench'] for window in train_windows]))
    mean = train_history.mean(dim=(0, 1), keepdim=True)
    scale = train_history.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    inputs = ((train_history - mean) / scale).flatten(1).to(device)
    targets = ((train_future - mean) / scale).flatten(1).to(device)
    validation_inputs = ((torch.from_numpy(np.stack([window['history'] for window in validation_windows])) - mean) / scale).flatten(1).to(device)
    validation_targets = ((torch.from_numpy(np.stack([window['future_wrench'] for window in validation_windows])) - mean) / scale).flatten(1).to(device)
    model = nn.Sequential(nn.Linear(60, 128), nn.SiLU(), nn.Linear(128, 1024), nn.SiLU(), nn.Linear(1024, 300)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        initial_loss = nn.functional.mse_loss(model(inputs), targets).item()
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(inputs), targets).item()
        validation_loss = nn.functional.mse_loss(model(validation_inputs), validation_targets).item()
    result = {'kind': 'auxiliary_wrench_mlp_smoke_NOT_VLA', 'device': device, 'steps': steps,
              'train_windows': len(train_windows), 'validation_windows': len(validation_windows),
              'train_episodes': [path.stem for path in files[:4]],
              'validation_episodes': [path.stem for path in files[-2:]],
              'initial_train_mse': initial_loss, 'final_train_mse': final_loss,
              'heldout_episode_mse': validation_loss,
              'images_or_language_used': False, 'pi05_loaded_or_trained': False,
              'normalization': 'training history windows only',
              'future_label_alignment': 'anchor through anchor+49; history is strictly causal'}
    output = ROOT / 'outputs' / f'force_smoke_{time.time_ns()}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect', 'isolation', 'environment', 'smoke'])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.command == 'inspect':
        inspect()
    elif args.command == 'environment':
        check_environment()
    elif args.command == 'isolation':
        print(json.dumps(isolation_report(), indent=2))
    else:
        smoke(args.steps, args.device)


if __name__ == '__main__':
    main()
