# tests/conftest.py
import pytest
import os
import sys

@pytest.fixture(scope='session', autouse=True)
def set_project_root():
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    if project_root not in sys.path:
        sys.path.append(project_root)
    os.chdir(project_root)
