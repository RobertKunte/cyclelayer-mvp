"""Empty package marker so `from tests.test_dataset import ...` works on all
platforms (Linux/Colab as well as Windows), regardless of pytest version
or rootdir auto-injection behaviour.

Adding this file makes `tests` an explicit Python package.  Combined with
the repo-root `conftest.py` and `pythonpath = ["."]` in pyproject.toml,
the absolute import in `test_pipeline_guards.py`
(`from tests.test_dataset import _make_synthetic_dataset`) resolves
deterministically.
"""
