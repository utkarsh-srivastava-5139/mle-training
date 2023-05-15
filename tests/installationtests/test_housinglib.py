import pytest


def test_library_installed():
    try:
        import housinglib_utkarsh
    except ImportError:
        pytest.fail("housinglib_utkarsh library is not installed")
