"""Mutation controls for the release gate's source and dataset contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from release_gate import canonical, inventory, negative_control, sha, verify_manifest


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'README.md').write_text('Reviewed documentation.\n')
        (self.root / 'program.py').write_text('print(1)\n')
        self.refresh()

    def refresh(self, data=False):
        files = inventory(self.root, {'manifest.json', 'SHA256SUMS'}, data=data)
        (self.root / 'manifest.json').write_text(json.dumps({'files': files, 'payload_sha256': canonical(files)}))
        if data:
            hashes = dict(files, **{'manifest.json': sha(self.root / 'manifest.json')})
            (self.root / 'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(hashes.items())))

    def test_readme_edit_requires_manifest_refresh(self):
        self.assertEqual(verify_manifest(self.root, 'manifest.json')['files'], 2)
        (self.root / 'README.md').write_text('Changed documentation.\n')
        with self.assertRaisesRegex(ValueError, 'Payload hash differs: README.md'):
            verify_manifest(self.root, 'manifest.json')
        self.refresh()
        verify_manifest(self.root, 'manifest.json')

    def test_unmanifested_source_is_rejected(self):
        (self.root / 'extra.py').write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verify_manifest(self.root, 'manifest.json')

    def test_payload_digest_cannot_be_changed(self):
        p = self.root / 'manifest.json'
        value = json.loads(p.read_text());value['payload_sha256'] = '0'*64;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'payload digest differs'):
            verify_manifest(self.root, 'manifest.json')

    def test_symlink_is_rejected_even_with_matching_bytes(self):
        (self.root / 'alias.py').symlink_to(self.root / 'program.py')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            verify_manifest(self.root, 'manifest.json')

    def test_unsafe_manifest_path_is_rejected(self):
        files = {'../outside': '0'*64}
        (self.root / 'manifest.json').write_text(json.dumps({'files': files, 'payload_sha256': canonical(files)}))
        with self.assertRaisesRegex(ValueError, 'Unsafe manifest path'):
            verify_manifest(self.root, 'manifest.json')

    def test_negative_control_preserves_original(self):
        before = (self.root / 'README.md').read_bytes()
        self.assertEqual(negative_control(self.root, 'manifest.json')['status'], 'passed')
        self.assertEqual((self.root / 'README.md').read_bytes(), before)

    def test_dataset_checksum_list_must_be_complete_and_unique(self):
        self.refresh(data=True)
        verify_manifest(self.root, 'manifest.json', data=True)
        p = self.root / 'SHA256SUMS';original = p.read_text()
        p.write_text(original.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, 'SHA256SUMS differs'):
            verify_manifest(self.root, 'manifest.json', data=True)
        p.write_text(original + original.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate checksum'):
            verify_manifest(self.root, 'manifest.json', data=True)

    def test_dataset_validation_outputs_are_not_silently_ignored(self):
        self.refresh(data=True)
        (self.root / 'validation').mkdir();(self.root / 'validation/result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            verify_manifest(self.root, 'manifest.json', data=True)


if __name__ == '__main__':
    unittest.main()
