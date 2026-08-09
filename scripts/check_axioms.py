#!/usr/bin/env python3
"""Axiom audit for the Lean formalization (CI gate).

Generates a #print axioms sweep over every top-level theorem/lemma in
PldrLlmMathFoundations/*.lean and fails if any result depends on an
axiom outside mathlib's standard three (propext, Classical.choice,
Quot.sound) or on sorryAx.  The project is kernel-checked relative to
ordinary classical principles -- not axiom-free in a foundational
sense -- and this gate keeps that statement true as dependencies move.

Usage (from the repo root, after `lake build`):

    python3 scripts/check_axioms.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "PldrLlmMathFoundations")
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}

names = []
for f in sorted(os.listdir(SRC)):
    if not f.endswith(".lean"):
        continue
    for line in open(os.path.join(SRC, f)):
        m = re.match(r"(?:theorem|lemma)\s+([A-Za-z0-9_']+)", line)
        if m:
            names.append("PldrLlm." + m.group(1))
if not names:
    sys.exit("axiom audit: no theorems found under PldrLlmMathFoundations/")

probe = "import PldrLlmMathFoundations\n" + "".join(
    f"#print axioms {n}\n" for n in names)
with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=ROOT,
                                 delete=False) as tf:
    tf.write(probe)
    path = tf.name
try:
    out = subprocess.run(["lake", "env", "lean", path], cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
finally:
    os.unlink(path)
if out.returncode != 0:
    print(out.stdout)
    print(out.stderr, file=sys.stderr)
    sys.exit("axiom audit: the probe file failed to elaborate")

checked = 0
bad = []
for line in out.stdout.splitlines():
    m = re.match(r"'([^']+)' depends on axioms: \[(.*)\]", line)
    if m:
        checked += 1
        axioms = {a.strip() for a in m.group(2).split(",") if a.strip()}
        extra = sorted(axioms - ALLOWED)
        if extra:
            bad.append((m.group(1), extra))
    elif "does not depend on any axioms" in line:
        checked += 1

print(f"axiom audit: {checked} declarations checked "
      f"({len(names)} probed); allowed: {sorted(ALLOWED)}")
if checked < len(names):
    print(out.stdout)
    sys.exit(f"axiom audit: only {checked} of {len(names)} probes "
             f"produced recognizable output")
if bad:
    for n, ax in bad:
        print(f"  FORBIDDEN axiom dependency in {n}: {ax}")
    sys.exit(1)
print("axiom audit: OK")
