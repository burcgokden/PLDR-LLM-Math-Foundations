#!/usr/bin/env python3
"""Download the immutable public inputs named in release-gate.json over HTTPS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True, help='A new directory outside the source checkout')
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.is_relative_to(ROOT):
        parser.error('Download inputs outside the source payload')
    dest.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'release-gate.json').read_text())
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    if config.get('dataset'):
        pin = config['dataset']
        data = dest / 'data'
        commands = [
            ['git', '-c', 'credential.helper=', 'clone', '--no-checkout', pin['url'], str(data)],
            ['git', '-C', str(data), 'lfs', 'install', '--local'],
            ['git', '-C', str(data), 'checkout', '--detach', pin['commit']],
            ['git', '-C', str(data), '-c', 'credential.helper=', 'lfs', 'pull', 'origin'],
            ['git', '-C', str(data), 'lfs', 'fsck'],
        ]
        for command in commands:
            subprocess.run(command, env=env, check=True, timeout=1800)
        manifest = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()
        if manifest != pin['manifest_sha256']:
            raise ValueError('Downloaded dataset manifest differs from the pinned release')
    request = urllib.request.Request(config['publication']['url'], headers={'User-Agent': 'PLDR-release-validation/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != config['publication']['sha256']:
        raise ValueError('Downloaded publication differs from the pinned PDF')
    (dest / 'publication.pdf').write_bytes(content)
    print('Downloaded pinned inputs to', dest)


if __name__ == '__main__':
    main()
