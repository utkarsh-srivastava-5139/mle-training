import pytest


def test_library_installed():
    try:
        import housinglib
    except ImportError:
        pytest.fail("housinglib library is not installed")
