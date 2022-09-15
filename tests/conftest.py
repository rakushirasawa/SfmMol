import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """
    $ pytest --markers
    """
    config.addinivalue_line("markers", "slow: mark for slow tests")


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="add --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
