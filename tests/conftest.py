"""Pytest configuration for CoCreator tests.

Registers --run-smoke flag to enable integration smoke tests.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="Run smoke tests (integration tests)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: mark test as a smoke/integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-smoke"):
        return
    skip_smoke = pytest.mark.skip(reason="need --run-smoke option to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)
