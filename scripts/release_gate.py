#!/usr/bin/env python3
"""Validate a release payload and write an execution record outside that payload.

The same standard-library gate is used by the three code companions. Configuration
is local and manifested. This command never stages, commits, tags or publishes.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.lake', 'build', 'validation', '__pycache__', '.pytest_cache', '.venv'}
SUFFIXES = {'.pyc', '.olean', '.ilean'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe manifest path: ' + name)
    target = Path(root).joinpath(*p.parts)
    if target.absolute() != target.resolve():
        raise ValueError('Symlink in payload: ' + name)
    return target


def inventory(root, excluded, data=False):
    ignored = {'.git'} if data else IGNORED
    files = {}
    for p in Path(root).rglob('*'):
        name = p.relative_to(root).as_posix()
        if set(p.relative_to(root).parts) & ignored or name in excluded:
            continue
        if not data and p.suffix in SUFFIXES:
            continue
        if p.is_symlink():
            raise ValueError('Symlink in payload: ' + name)
        if p.is_file():
            files[name] = sha(p)
    return dict(sorted(files.items()))


def verify_manifest(root, name, data=False):
    excluded = {name, 'SHA256SUMS'} if data else {name}
    path = local(root, name)
    manifest = json.loads(path.read_text())
    expected = manifest['files']
    for item in expected:
        local(root, item)
    if canonical(expected) != manifest['payload_sha256']:
        raise ValueError('Manifest payload digest differs: ' + name)
    actual = inventory(root, excluded, data)
    if set(actual) != set(expected):
        raise ValueError('Payload inventory differs: missing=' + repr(sorted(set(expected)-set(actual))[:5])
                         + '; unexpected=' + repr(sorted(set(actual)-set(expected))[:5]))
    changed = [n for n in expected if expected[n] != actual[n]]
    if changed:
        raise ValueError('Payload hash differs: ' + ', '.join(changed[:5]))
    if data:
        sums = {}
        for line in (root / 'SHA256SUMS').read_text().splitlines():
            digest, item = line.split('  ', 1)
            if item in sums:
                raise ValueError('Duplicate checksum path: ' + item)
            local(root, item)
            sums[item] = digest
        if sums != dict(actual, **{name: sha(path)}):
            raise ValueError('SHA256SUMS differs from the complete dataset payload and manifest')
    return {'manifest': name, 'manifest_sha256': sha(path),
            'payload_sha256': canonical(actual), 'files': len(actual)}


def negative_control(root, name, data=False):
    manifest = json.loads((root / name).read_text())
    with tempfile.TemporaryDirectory(prefix='pldr-readme-negative-') as tmp:
        copy = Path(tmp)
        for item in [*manifest['files'], name, *(['SHA256SUMS'] if data else [])]:
            target = copy / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / item, target)
        with (copy / 'README.md').open('a') as stream:
            stream.write('\nIntentional stale-manifest negative control.\n')
        try:
            verify_manifest(copy, name, data)
        except ValueError as exc:
            if 'Payload hash differs: README.md' not in str(exc):
                raise
            return {'status': 'passed', 'rejected': str(exc)}
    raise ValueError('README mutation was incorrectly accepted')


def git_identity(root):
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        if p.returncode:
            raise ValueError('Cannot establish Git identity: ' + ' '.join(args))
        return p.stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'),
            'status': git('status', '--porcelain=v1', '--untracked-files=all'),
            'scope': 'HEAD is a base identity until the working tree is clean.'}


def xml_results(paths):
    result = []
    for path in paths:
        cases = ET.parse(path).getroot().findall('.//testcase')
        result.append({'file': str(path.relative_to(ROOT)), 'cases': len(cases),
                       'skips': [{'test': c.attrib, 'reason': c.find('skipped').get('message', '')}
                                 for c in cases if c.find('skipped') is not None]})
    return result


def execute(config, args, record, out):
    identity = git_identity(ROOT)
    record['source'] = identity
    if args.require_clean and identity['status']:
        raise ValueError('Release requires a clean committed checkout; candidate edits are still present')
    record['candidate'] = bool(identity['status']) or args.candidate_data
    record['code'] = verify_manifest(ROOT, config['manifest'])
    record['negative_controls'] = {'code_readme': negative_control(ROOT, config['manifest'])}
    pdf = args.pdf.resolve()
    record['publication'] = {'url': config['publication']['url'], 'sha256': sha(pdf)}
    if record['publication']['sha256'] != config['publication']['sha256']:
        raise ValueError('PDF hash differs from the bound publication artifact')
    data = None
    if config.get('dataset'):
        if args.data_repo is None:
            raise ValueError('--data-repo is required for this companion')
        data = args.data_repo.resolve()
        record['dataset'] = verify_manifest(data, 'manifest.json', data=True)
        record['dataset']['source'] = git_identity(data) if (data / '.git').exists() else {'commit': None, 'status': 'export'}
        record['dataset']['index_sha256'] = sha(data / 'index.json')
        binding = config['dataset']
        if record['dataset']['index_sha256'] != binding['index_sha256']:
            raise ValueError('Dataset index differs from the supported scientific records')
        if not args.candidate_data:
            for key in ['manifest_sha256', 'payload_sha256']:
                if record['dataset'][key] != binding[key]:
                    raise ValueError('Dataset '+key+' differs from the pinned release; use --candidate-data only for reviewed local preparation')
            source = record['dataset']['source']
            if source['commit'] is not None and (source['commit'] != binding['commit'] or source['status']):
                raise ValueError('Dataset checkout must be clean at the configured full commit')
        record['negative_controls']['dataset_readme'] = negative_control(data, 'manifest.json', data=True)
    tokens = {'{python}': sys.executable, '{pdf}': str(pdf), '{data}': str(data)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    record['checks'] = []
    generated_xml = []
    for job in config['checks']:
        argv = [tokens.get(value, value) for value in job['argv']]
        cwd = ROOT if job.get('cwd', '.') == '.' else local(ROOT, job['cwd'])
        before_xml = {p: (p.stat().st_mtime_ns, p.stat().st_size)
                      for p in ROOT.rglob('*.xml') if 'validation' in p.relative_to(ROOT).parts and '.git' not in p.parts}
        log = out.parent / (job['name'] + '.log')
        start = time.monotonic()
        print('RUN', job['name'], flush=True)
        with log.open('w') as stream:
            p = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=job.get('timeout_seconds', 1800))
        row = {'name': job['name'], 'argv': argv, 'cwd': job.get('cwd', '.'),
               'returncode': p.returncode, 'seconds': time.monotonic()-start,
               'log': log.name, 'log_sha256': sha(log)}
        record['checks'].append(row)
        for path in ROOT.rglob('*.xml'):
            if 'validation' not in path.relative_to(ROOT).parts or '.git' in path.parts:
                continue
            if before_xml.get(path) != (path.stat().st_mtime_ns, path.stat().st_size):
                generated_xml.append(path)
        if p.returncode:
            raise ValueError('Failed check: '+job['name']+'; see '+str(log))
    record['scientific_suites'] = xml_results(sorted(set(generated_xml)))
    if verify_manifest(ROOT, config['manifest']) != record['code']:
        raise ValueError('Source payload changed during validation')
    if git_identity(ROOT) != identity:
        raise ValueError('Source Git identity changed during validation')
    if data and verify_manifest(data, 'manifest.json', data=True) != {k: record['dataset'][k] for k in ['manifest', 'manifest_sha256', 'payload_sha256', 'files']}:
        raise ValueError('Dataset changed during validation')
    if sha(pdf) != record['publication']['sha256']:
        raise ValueError('Publication artifact changed during validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-repo', type=Path)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--candidate-data', action='store_true', help='Validate local metadata edits without claiming the configured dataset release')
    parser.add_argument('--require-clean', action='store_true', help='Require the tested source to be a committed, clean checkout')
    parser.add_argument('--refresh-manifest', action='store_true', help='Explicitly record reviewed source bytes; does not run tests')
    parser.add_argument('--output', type=Path, default=Path('validation/release-gate.json'))
    args = parser.parse_args()
    config = json.loads((ROOT / 'release-gate.json').read_text())
    if args.refresh_manifest:
        name = config['manifest']
        if config.get('book'):
            parser.error('Refresh nested/book provenance with scripts/update_release_manifest.py instead')
        path = ROOT / name
        manifest = json.loads(path.read_text()) if path.exists() else {'schema': 'pldr-source-files-v1'}
        manifest['files'] = inventory(ROOT, {name})
        manifest['payload_sha256'] = canonical(manifest['files'])
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
        print('Recorded reviewed bytes:', name, manifest['payload_sha256'])
        return 0
    if args.pdf is None:
        parser.error('--pdf is required')
    if args.candidate_data and args.require_clean:
        parser.error('--candidate-data and --require-clean are mutually exclusive')
    out = args.output.absolute()
    if out.is_relative_to(ROOT) and 'validation' not in out.relative_to(ROOT).parts:
        parser.error('Write execution records under ignored validation/ or outside the repository')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {'schema': 'pldr-release-execution-v1',
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])),
              'github_run': {name: os.environ[name] for name in ['GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REF'] if name in os.environ},
              'scope': 'Source integrity, CPU scientific checks and bound publication/evidence checks. Lean and GPU results are separate; no new training campaign.'}
    try:
        execute(config, args, record, out)
        record['status'] = 'passed'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = type(exc).__name__+': '+str(exc)
    record['seconds'] = time.monotonic()-started
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': record['status'], 'output': str(out), 'error': record.get('error')}))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
