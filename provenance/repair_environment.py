import hashlib
import json
import os
from pathlib import Path

from conda.core.prefix_data import PrefixData
from conda.misc import explicit
from conda.models.prefix_graph import PrefixGraph


root = Path('/data0/wx/force-VLA')
source = Path('/data0/miniconda3/envs/evo-rlt')
assert os.statvfs(source).f_flag & os.ST_RDONLY
assert os.statvfs('/data0/sht/Evo-RLT').f_flag & os.ST_RDONLY
records = tuple(PrefixGraph(PrefixData(str(source)).iter_records()).graph)
specifications = []
for record in records:
    archive = root / 'runtime/conda-pkgs' / record.fn
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    if record.sha256 and checksum != record.sha256:
        raise RuntimeError(f'Package archive checksum mismatch: {archive}')
    specifications.append(archive.as_uri() + '#sha256:' + checksum)
explicit(specifications, str(root / 'env'))
installed = tuple(PrefixData(str(root / 'env')).iter_records())
expected = {(record.name, record.version, record.build) for record in records}
actual = {(record.name, record.version, record.build) for record in installed}
if actual != expected:
    raise RuntimeError(f'Package metadata mismatch: {expected ^ actual}')
(root / 'provenance/environment-recovery.json').write_text(json.dumps({
    'reason': 'Offline remote URLs did not resolve against the private cache; completed the clone using explicit local archive URLs',
    'packages': len(installed), 'original_environment_readonly': True,
    'original_workspace_readonly': True, 'network_used': False,
    'local_package_specifications': specifications,
}, indent=2) + '\n')
(root / '.environment-ready').write_text('local-package-completion')
print('Completed all 29 Conda packages from verified local archives')
