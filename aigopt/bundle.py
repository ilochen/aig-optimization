"""Portable result provenance and independent checks; never execute trace commands."""
import json
import math
from pathlib import Path, PurePosixPath
import platform
import shutil
import tempfile
from . import __version__
from .abc import ABC, fingerprint
from .graph import Graph
from .objective import Objective

SCHEMA = 'aigopt-result-v1'
REQUIRED = {'final.aig', 'final.mapped.blif', 'reference.aig', 'model.json', 'trace.json'}


def write_bundle(staged, *, reference, source, library, model, backend, settings):
    staged = Path(staged)
    suffix = Path(source).suffix.lower()
    if suffix not in ('.aig', '.aiger', '.blif', '.truth', '.v', '.bench', '.pla'):
        raise ValueError('Unsupported source suffix for a portable result bundle')
    source_name = 'source' + suffix
    library_name = 'library.lib' if backend.liberty else 'library.genlib'
    for origin, name in ((reference, 'reference.aig'), (source, source_name),
                         (library, library_name), (model, 'model.json')):
        shutil.copyfile(origin, staged / name)
    files = {p.relative_to(staged).as_posix(): fingerprint(p)
             for p in sorted(staged.rglob('*')) if p.is_file()}
    manifest = {'schema': SCHEMA, 'source': source_name, 'library': library_name, 'files': files,
        'provenance': {'aigopt_version': __version__, 'python': platform.python_version(),
            'platform': platform.platform(), 'abc_sha256': fingerprint(backend.executable),
            'implementation_sha256': {p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob('*.py'))}},
        'settings': settings,
        'verification_note': 'Hashes detect changes; they are not signatures. Recheck equivalence with aigopt verify.'}
    (staged / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def check_bundle(directory):
    """Reject malformed inventories, path escapes, missing files, and corruption."""
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported result bundle schema')
    files = manifest.get('files')
    source = manifest.get('source')
    if not isinstance(files, dict) or not isinstance(source, str) or not source.startswith('source.'):
        raise ValueError('Invalid artifact inventory')
    library_name = manifest.get('library', 'library.genlib')
    if library_name not in ('library.genlib', 'library.lib'):
        raise ValueError('Invalid library artifact name')
    if not REQUIRED.union({source, library_name}).issubset(files) or 'manifest.json' in files:
        raise ValueError('Incomplete artifact inventory')
    for name, digest in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name:
            raise ValueError('Unsafe artifact path')
        target = root / name
        if not target.is_file() or target.is_symlink() or root not in target.resolve().parents:
            raise ValueError(f'Missing or unsafe artifact: {name}')
        if not isinstance(digest, str) or fingerprint(target) != digest:
            raise ValueError(f'Artifact hash mismatch: {name}')
    trace = json.loads((root / 'trace.json').read_text())
    for field, name in (('input_sha256', source), ('model_sha256', 'model.json'), ('library_sha256', library_name)):
        if trace.get(field) != files[name]:
            raise ValueError(f'Trace provenance mismatch: {field}')
    return root, manifest, trace


def _match_metrics(actual, claimed, name):
    for metric in ('area', 'delay'):
        expected = claimed[metric]
        if not math.isfinite(expected) or expected <= 0 or not math.isclose(actual[metric], expected, rel_tol=1e-9, abs_tol=.0051):
            raise ValueError(f'{name} {metric} mismatch: measured {actual[metric]}, reported {expected}')


def verify_bundle(directory, abc, *, timeout=60, allow_different_abc=False):
    """Recheck source → reference → final AIG/netlist, costs, and acceptance.

    Uses a private workspace, so verification never modifies the supplied bundle.
    The checkpoint and trace are evidence only; no saved commands are executed.
    """
    root, manifest, trace = check_bundle(directory)
    same_tool = fingerprint(abc) == manifest['provenance']['abc_sha256']
    if not same_tool and not allow_different_abc:
        raise ValueError('ABC executable differs; use --allow-different-abc for an explicit revalidation')
    objective = Objective(**trace['objective'])
    with tempfile.TemporaryDirectory(prefix='aig-recheck-') as directory:
        work = Path(directory)
        library_name = manifest.get('library', 'library.genlib')
        library = work / ('cells.lib' if library_name.endswith('.lib') else 'cells.genlib')
        shutil.copyfile(root / library_name, library)
        backend = ABC(abc, library, timeout=timeout)
        backend.validate_profile(trace)
        original = backend.prepare(root / manifest['source'], work / 'source.aig')
        reference = work / 'reference.aig'; final = work / 'final.aig'
        shutil.copyfile(root / 'reference.aig', reference)
        shutil.copyfile(root / 'final.aig', final)
        netlist = work / 'final.mapped.blif'
        shutil.copyfile(root / 'final.mapped.blif', netlist)
        for target in (reference, final, netlist):
            backend.verify(original, target)
        baseline = backend.map(reference)
        remapped = backend.map(final)
        submitted = backend.measure(netlist)
        _match_metrics(baseline, trace['baseline_mapped'], 'Baseline')
        _match_metrics(remapped, trace['final_mapped'], 'Remapped final')
        _match_metrics(submitted, trace['final_mapped'], 'Exported netlist')
        if objective.key(remapped) > objective.key(baseline):
            raise ValueError('Exported result regresses against the baseline under its objective')
        if objective.describe(remapped) != trace['objective_result']:
            raise ValueError('Reported feasibility or objective cost does not match measured result')
        if Graph.read(final).features() != trace['final_features']:
            raise ValueError('Reported final graph features do not match')
    return {'bundle_valid': True, 'equivalence_proven': True, 'metrics_reproduced': True,
            'same_abc_binary': same_tool, 'objective': objective.payload(),
            'objective_result': objective.describe(remapped), 'mapping_calls': backend.mapping_calls,
            'verification_calls': backend.verification_calls}
