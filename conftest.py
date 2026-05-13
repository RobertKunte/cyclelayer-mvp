"""Repo-root conftest — guarantees pytest rootdir + sys.path on all platforms.

Without this file, pytest's rootdir detection differs between platforms:
on Windows it picks up the repo root via `pyproject.toml` and adds it to
`sys.path`, but on Linux/Colab containers the absolute import
`from tests.test_dataset import _make_synthetic_dataset` in
`tests/test_pipeline_guards.py` fails with `ModuleNotFoundError: No module
named 'tests.test_dataset'`.

Adding this empty conftest.py at the repo root makes pytest treat this
directory as the rootdir and inject it into `sys.path` before collecting
tests — fixing the Colab failure without touching any test file.

References:
- https://docs.pytest.org/en/stable/reference/customize.html#rootdir
- https://docs.pytest.org/en/stable/explanation/pythonpath.html
"""

# No code needed — the mere existence of this file is what pytest needs.
"""anchor"""
