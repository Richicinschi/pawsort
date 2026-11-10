import os

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import create_app

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def photos(tmp_path):
    """The folder where the orchestrator of a test keeps its photos."""
    return tmp_path / "photos"


@pytest.fixture
def orch_client(tmp_path, photos):
    """A real orchestrator on a temporary database, reached over its HTTP API."""
    return TestClient(create_app(str(tmp_path / "jobs.db"), str(photos)))


@pytest.fixture
def work(tmp_path):
    """An empty working folder for an AI server."""
    folder = tmp_path / "work"
    folder.mkdir()
    return str(folder)


@pytest.fixture
def send_photo():
    """Send a test photo to an orchestrator the way the website does. Gives back the job number."""
    def _send(orch, name: str = "bobcat_1.jpg", site_code: str | None = None) -> str:
        with open(os.path.join(DATA, name), "rb") as f:
            r = orch.post("/jobs", data={"site_code": site_code} if site_code else {},
                          files={"photo": ("photo", f.read(), "image/jpeg")})
        return r.json()["id"]
    return _send
