# Building and Testing a Local sdist

This document describes how to build an `openfold3` sdist/wheel locally and
verify it installs and runs correctly. This is a pre-release sanity check —
it does **not** cover tagging, versioning, or publishing. Actual releases go
through the release pipeline once local testing here passes.

## 1. Build in a clean environment

```bash
cd ~/workspace/openfold-3
# make sure you're on the branch/commit you want to test

mamba create -n dev python=3.14
conda activate dev

python3 -m build
```

This produces a `dist/openfold3-<version>.tar.gz` (sdist) and a
`dist/openfold3-<version>-py3-none-any.whl` (wheel) in the repo's `dist/`
directory. Since the working tree isn't tagged for a real release,
`setuptools_scm` will typically generate a local dev version like
`0.4.2.dev146+gf4063eb56.d20260701` — that's expected and fine for this kind
of local testing.

Optionally check the sdist metadata is well-formed before installing it:

```bash
twine check dist/*
```

## 2. Install the sdist into a fresh venv with uv

Testing from a clean venv (rather than your dev environment) is what
actually catches packaging bugs — missing files, incorrect package data,
wrong dependency pins — that `python -m build` alone won't surface.

```bash
mkdir test-sdist && cd test-sdist
uv venv
source .venv/bin/activate

uv pip install ../dist/openfold3-<version>.tar.gz[dev]
```

(Adjust the path/version to match whatever landed in `dist/`.)

## 3. Run the test suite against the installed package

```bash
pytest --pyargs openfold3
```

Running with `--pyargs` is important here — it runs tests against the
**installed** package in `.venv/`, not against the source tree, which is the
point of this exercise (confirming the packaged artifact actually works).

## 4. (Optional) Repeat against the wheel

The sdist and wheel can diverge — e.g. a file missing from `MANIFEST.in`
might only affect the sdist, while a stale `build` isolation issue might only
show up in the wheel. If you want to be thorough, repeat steps 2–3 with
`dist/openfold3-<version>-py3-none-any.whl` in a separate fresh venv.

---

## References

- [PyPA: Creating and using virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) — background on why installing into an isolated environment (rather than testing in-place) is standard practice before a release.
- [`build` documentation](https://build.pypa.io/) — details on what `python -m build` does and how it isolates the build from your dev environment.
- [`setuptools_scm` documentation](https://setuptools-scm.readthedocs.io/) — explains the `.devN+gHASH.dDATE` version scheme you'll see on untagged commits.
- [pytest: Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html) — covers the rationale for `--pyargs` / testing installed packages vs. the source tree.
