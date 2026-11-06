import os
import shutil

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import create_app

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def orch_client(tmp_path):
    """A real orchestrator on a temporary database, reached over its HTTP API."""
    return TestClient(create_app(str(tmp_path / "jobs.db")))


@pytest.fixture
def copy_photo(tmp_path):
    """Copy a test photo into the temp folder (the AI server deletes photos after use)."""
    def _copy(name: str) -> str:
        dst = tmp_path / name
        shutil.copy(os.path.join(DATA, name), dst)
        return str(dst.resolve())
    return _copy
