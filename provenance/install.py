from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import time


SOURCE = Path('/data0/sht/Evo-RLT')
SOURCE_ENV = Path('/data0/miniconda3/envs/evo-rlt')
DESTINATION = Path('/data0/wx/force-VLA')
STAGING = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy_verified(source, target, records):
    source = source.resolve(strict=True)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size > before.st_size:
            raise RuntimeError(f'Unexpected existing target: {target}')
        with source.open('rb') as original, target.open('rb') as partial:
            for block in iter(lambda: partial.read(8 * 1024 * 1024), b''):
                if block != original.read(len(block)):
                    raise RuntimeError(f'Existing copy differs from source: {target}')
            with target.open('ab') as output:
                shutil.copyfileobj(original, output, 8 * 1024 * 1024)
        shutil.copystat(source, target)
    else:
        shutil.copy2(source, target)
    source_hash = digest(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed during copy: {source}')
    if source_hash != digest(target):
        raise RuntimeError(f'Copy hash mismatch: {target}')
    if (source.stat().st_dev, source.stat().st_ino) == (target.stat().st_dev, target.stat().st_ino):
        raise RuntimeError(f'Shared inode: {target}')
    records.append({'source': str(source), 'destination': str(target.relative_to(DESTINATION)),
                    'bytes': before.st_size, 'source_mtime_ns': before.st_mtime_ns,
                    'sha256': source_hash})


def copy_tree(source, target, records, code_only=False):
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if any(part in {'.git', '__pycache__', '.pytest_cache', '.ruff_cache'} for part in relative.parts):
            continue
        if path.is_file():
            if code_only and path.suffix not in {'.py', '.yaml', '.yml', '.json', '.md', '.sh', '.toml', '.urdf', '.xml'}:
                continue
            copy_verified(path, target / relative, records)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def clean_environment():
    runtime = DESTINATION / 'runtime'
    return {'PATH': '/usr/bin:/bin', 'HOME': str(runtime / 'home'), 'LANG': 'C.UTF-8',
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'CONDA_PKGS_DIRS': str(runtime / 'conda-pkgs'),
            'CONDA_ENVS_PATH': str(runtime / 'conda-envs'),
            'CONDA_NO_PLUGINS': 'true', 'CONDA_SOLVER': 'classic',
            'CONDA_OFFLINE': 'true', 'CONDA_AUTO_UPDATE_CONDA': 'false',
            'CONDARC': str(runtime / 'condarc'),
            'XDG_CACHE_HOME': str(runtime / 'cache'),
            'XDG_CONFIG_HOME': str(runtime / 'config'),
            'XDG_DATA_HOME': str(runtime / 'data'), 'TMPDIR': str(runtime / 'tmp')}


def main():
    if not os.statvfs(SOURCE).f_flag & os.ST_RDONLY:
        raise RuntimeError('Installer must run with the source filesystem mounted read-only')
    if not os.statvfs(SOURCE_ENV).f_flag & os.ST_RDONLY:
        raise RuntimeError('Original environment must be mounted read-only')
    marker = DESTINATION / '.migration-in-progress'
    if list(DESTINATION.iterdir()) and (not marker.is_file() or marker.read_text() != str(SOURCE)):
        raise RuntimeError('Destination must be empty or contain our verified migration marker')
    marker.write_text(str(SOURCE))
    if shutil.disk_usage(DESTINATION).free < 15 * 1024**3:
        raise RuntimeError('Need at least 15 GiB free for the remaining copies and environment')
    started = time.time()
    records = []
    for name in ['provenance', 'outputs', 'annotations', 'configs', 'runtime/home',
                 'runtime/cache', 'runtime/config', 'runtime/data', 'runtime/tmp',
                 'runtime/conda-pkgs', 'runtime/conda-envs', 'src', 'tests', 'scripts']:
        (DESTINATION / name).mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'runtime/condarc').write_text('offline: true\nauto_update_conda: false\n')
    for name in ['franka_single_left_rgb_block', 'gamepad_task3_20260909_online_c20_full',
                 'franka_single_left_gamepad_task3__raw']:
        print(f'Copying raw dataset: {name}', flush=True)
        copy_tree(SOURCE / 'datasets/raw' / name, DESTINATION / 'datasets/raw' / name, records)
    split_name = 'franka_single_left_gamepad_task3_20260908_split_seed1000'
    copy_tree(SOURCE / 'datasets/sft' / split_name, DESTINATION / 'datasets/sft_reference' / split_name, records)
    sft = SOURCE / 'outputs/sft/franka_single_left_gamepad_pi05_sft_task3_20260908_clean/checkpoints/073026/pretrained_model'
    print('Copying PI05 SFT checkpoint', flush=True)
    copy_tree(sft, DESTINATION / 'models/pi05_sft', records)
    token = SOURCE / 'outputs/rltoken/gamepad_task3_20260908_rl_token/checkpoints/010000/pretrained_model'
    copy_tree(token, DESTINATION / 'models/rl_token_reference', records)
    for name in ['tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'config.json', 'preprocessor_config.json']:
        copy_verified(SOURCE / 'models/paligemma' / name, DESTINATION / 'models/tokenizer' / name, records)
    run = SOURCE / 'outputs/ac/gamepad_task3_20260909_online_c20_full'
    copy_verified(run / 'replay.sqlite3', DESTINATION / 'reference/rlt/replay.sqlite3', records)
    copy_tree(run / 'checkpoints/checkpoint-1789021365979913376',
              DESTINATION / 'models/ac_reference', records)
    for name in ['src', 'tests', 'scripts', 'datacollection']:
        copy_tree(SOURCE / name, DESTINATION / 'reference/evo-rlt' / name, records, code_only=True)
    for name in ['README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock']:
        copy_verified(SOURCE / name, DESTINATION / 'reference/evo-rlt' / name, records)
    paper = Path('/tmp/evo-rlt-2509.07962.html')
    if paper.is_file():
        copy_verified(paper, DESTINATION / 'reference/paper/2509.07962v1.html', records)
    write_json(DESTINATION / 'provenance/assets.json', records)
    environment_before = {str(path.relative_to(SOURCE_ENV)): digest(path)
                          for path in sorted((SOURCE_ENV / 'conda-meta').glob('*')) if path.is_file()}
    editable = SOURCE_ENV / 'lib/python3.12/site-packages/__editable__.evo_rlt-0.1.0.pth'
    environment_before[str(editable.relative_to(SOURCE_ENV))] = digest(editable)
    for path in sorted((SOURCE_ENV / 'conda-meta').glob('*.json')):
        record = json.loads(path.read_text())
        archive = Path('/data0/miniconda3/pkgs') / record['fn']
        shutil.copy2(archive, DESTINATION / 'runtime/conda-pkgs' / archive.name)
    print('Cloning environment offline, with independent file copies and private caches', flush=True)
    command = ['/data0/miniconda3/bin/conda', 'create', '--prefix', str(DESTINATION / 'env'),
               '--clone', str(SOURCE_ENV), '--copy', '--offline', '--no-default-packages',
               '--no-shortcuts', '--yes']
    if not (DESTINATION / '.environment-ready').exists():
        with (DESTINATION / 'provenance/conda-clone.log').open('w') as log:
            subprocess.run(command, env=clean_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    site = DESTINATION / 'env/lib/python3.12/site-packages'
    (site / editable.name).write_text(str(DESTINATION / 'reference/evo-rlt/src') + '\n')
    write_json(site / 'evo_rlt-0.1.0.dist-info/direct_url.json',
               {'dir_info': {'editable': True}, 'url': (DESTINATION / 'reference/evo-rlt').as_uri()})
    external_links = []
    shared_files = []
    env_files = 0
    for path in (DESTINATION / 'env').rglob('*'):
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(DESTINATION) and not str(target).startswith(('/usr/', '/lib/')):
                external_links.append([str(path), str(target)])
        elif path.is_file():
            env_files += 1
            original = SOURCE_ENV / path.relative_to(DESTINATION / 'env')
            if original.exists() and (path.stat().st_dev, path.stat().st_ino) == (original.stat().st_dev, original.stat().st_ino):
                shared_files.append(str(path))
    if external_links or shared_files:
        raise RuntimeError(f'Environment not independent: links={external_links}, shared={shared_files[:10]}')
    for path in site.glob('*.pth'):
        if str(SOURCE) in path.read_text(errors='replace') or '/opt/ros' in path.read_text(errors='replace'):
            raise RuntimeError(f'Contaminated import path: {path}')
    write_json(DESTINATION / 'provenance/python-packages.json',
               dict(sorted((distribution.metadata['Name'], distribution.version)
                           for distribution in importlib.metadata.distributions(path=[str(site)])
                           if distribution.metadata['Name'])))
    for name in ['run.sh', 'lab.py', 'test_lab.py', 'README.md', 'AGENTS.md']:
        target = {'run.sh': 'scripts/run.sh', 'lab.py': 'src/lab.py', 'test_lab.py': 'tests/test_lab.py'}.get(name, name)
        shutil.copy2(STAGING / name, DESTINATION / target)
    shutil.copy2(STAGING / 'install.py', DESTINATION / 'provenance/install.py')
    (DESTINATION / 'scripts/run.sh').chmod(0o755)
    write_json(DESTINATION / 'configs/experiment.json', {
        'status': 'prepared_not_force_vla_trained', 'policy_checkpoint': 'models/pi05_sft',
        'tokenizer': 'models/tokenizer', 'history_seconds': 2.0, 'history_frames': 10,
        'wrench_state_slice': [10, 16], 'action_dim': 10, 'future_steps': 50,
        'force_loss_weight_proposal': 0.1, 'history_token_dim_proposal': 1024,
        'datasets': {'sft_100': 'datasets/raw/franka_single_left_rgb_block',
                     'rlt_full_113': 'datasets/raw/gamepad_task3_20260909_online_c20_full',
                     'prior_sft_102': 'datasets/raw/franka_single_left_gamepad_task3__raw'},
        'rlt_80_subset': None, 'sft_100_task_text': '111',
        'annotation_status': 'requires_human_confirmed_language_labels',
        'copied_checkpoint_configs': 'unaltered historical records; use explicit local path overrides',
        'force_model_status': 'not_implemented; smoke test is auxiliary MLP only'})
    for record in records:
        path = Path(record['source'])
        if digest(path) != record['sha256'] or path.stat().st_mtime_ns != record['source_mtime_ns']:
            raise RuntimeError(f'Source changed: {path}')
    for relative, expected in environment_before.items():
        if digest(SOURCE_ENV / relative) != expected:
            raise RuntimeError(f'Original environment metadata changed: {relative}')
    write_json(DESTINATION / 'provenance/migration.json', {
        'source_repository': str(SOURCE), 'source_environment': str(SOURCE_ENV),
        'source_commit': subprocess.check_output(['/usr/bin/git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'approved_remote': 'https://github.com/MINT-SJTU/Evo-RLT',
        'destination': str(DESTINATION), 'copied_asset_files': len(records),
        'copied_asset_bytes': sum(record['bytes'] for record in records),
        'environment_regular_files_checked_for_shared_inodes': env_files,
        'environment_shared_source_inodes': shared_files, 'environment_external_links': external_links,
        'source_assets_hash_and_mtime_unchanged': True,
        'source_environment_metadata_unchanged': True,
        'source_and_original_environment_mounted_readonly_during_migration': True,
        'environment_metadata_sha256': environment_before, 'seconds': time.time() - started,
        'base_and_paligemma_full_weights_omitted': 'SFT weights provide backbone; avoid redundant 25 GB copy',
        'complete': True})
    marker.unlink()
    print(f'Migration complete: {DESTINATION}', flush=True)


if __name__ == '__main__':
    main()
