# Release validation and identity

The CPU release gate authenticates the complete source inventory and canonical
payload digest, rejects a stale README hash in a disposable copy, runs the
maintained CPU suites and checks the exact publication PDF bound in
`release-gate.json`. Dynamics and Book Companion also authenticate all compact
evidence, reject a stale dataset README hash, check code/data correspondence,
and resolve the monograph's 263 PDF destinations. The Foundations PDF check
authenticates its bytes; it does not check paper destinations or proofs.

## Run the gate

Install the existing pinned CPU check dependencies first. Use Python 3.14.6
for the tested scientific stack; the gate and downloader use the standard
library. The download also requires Git and Git LFS. A fresh anonymous download
can be prepared outside the code checkout:

```sh
python3 scripts/download_release_inputs.py --destination ../pldr-release-inputs
```

Use `../pldr-release-inputs/publication.pdf` for `PUBLICATION.pdf` and, where
applicable, `../pldr-release-inputs/data` for `DATA-REPO` below:

```sh
python3 scripts/release_gate.py --pdf PUBLICATION.pdf
```

The record `validation/release-gate.json` includes UTC timestamps, base Git
commit and working-tree status, payload and manifest hashes, publication/data
identities, observed package versions, commands, return codes, log hashes and
named fixture skips from freshly generated JUnit reports. Records and logs stay
outside the authenticated payload to avoid a self-referential hash. A dirty
checkout is explicitly a candidate; its base HEAD is not its release commit.

For reviewed local dataset metadata edits, add `--candidate-data` in the Dynamics
or Book gate. That mode still verifies every byte, checksum, index and mapping,
but records the actual dataset identity instead of claiming the configured
published revision. It cannot be combined with `--require-clean`.

## Review, refresh and publish a release

1. Review the source changes and preserve historical records and scientific data.
2. Refresh reviewed source hashes with `python3 scripts/release_gate.py --refresh-manifest`. In the Book Companion,
   the existing updater refreshes the nested Dynamics manifest and book provenance.
3. Run the gate against the final bytes and inspect the execution record and skips.
4. The maintainer decides the commits. If dataset metadata is released, commit it
   first, then put its full commit, manifest-file and payload hashes in each
   consuming `release-gate.json`; retain the unchanged scientific index binding.
   Refresh source manifests again after this configuration edit.
5. After committing the reviewed source, rerun the gate with `--require-clean`.
   This produces a record bound to the final commit; a pre-commit candidate pass
   does not certify that future commit.
6. Select a new version tag for that tested commit. Preserve old tags and the
   book's Appendix C pin. Attach the final JSON record and logs to the release;
   the Actions run also retains them for 90 days. No script here creates a tag,
   commit or release automatically.

The workflow `.github/workflows/release-validation.yml` runs on pull requests,
main-branch pushes, version tags and manual dispatch. It installs the declared
dependencies, downloads immutable public inputs, requires a clean source
checkout, and uploads the execution record even when a check fails. CI never
refreshes hashes to make a failing payload pass.

## Separate validation scopes

Lean compilation and axiom checks remain separately named formal checks using
the pinned toolchain. Foundations retains its existing formal CI job; Dynamics
and Book provide an optional `run_lean` job on manual release-workflow dispatch.
GPU smoke and full model/data acquisition require their declared external assets
and remain separate from the CPU release gate. CPU success does not imply a
fresh Lean build, a GPU result, or reproduction of the training campaigns.

Historical validation records retain their original counts and context. New
execution artifacts establish only the named checks on the recorded payload.
