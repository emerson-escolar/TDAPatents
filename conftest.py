import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--comprehensive", action="store_true", default=False, help="run comprehensive tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", ": mark test as part of comprehensive testing")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--comprehensive"):
        return
    skip_compre = pytest.mark.skip(reason="need --comprehensive option to run")
    for item in items:
        if "comprehensive" in item.keywords:
            item.add_marker(skip_compre)
