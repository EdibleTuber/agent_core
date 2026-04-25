"""Smoke test: agent_core package imports cleanly."""


def test_package_imports():
    import agent_core
    assert agent_core is not None


def test_utils_subpackage_imports():
    from agent_core import utils
    assert utils is not None
