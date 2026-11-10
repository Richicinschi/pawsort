import os

import pytest

from orchestrator.app import delete_photo


@pytest.fixture
def photo(tmp_path):
    photo = tmp_path / "abc.jpg"
    photo.write_bytes(b"photo")
    return photo


def busy(times, calls):
    """An os.remove that says the file is in use for the first few calls, like Windows while somebody reads it."""
    real = os.remove

    def remove(path):
        calls.append(path)
        if len(calls) <= times:
            raise PermissionError("the file is in use")
        real(path)
    return remove


def test_deleting_a_photo_removes_the_file(photo):
    assert delete_photo(str(photo)) is True
    assert not photo.exists()


def test_deleting_a_photo_that_is_gone_is_fine(tmp_path):
    assert delete_photo(str(tmp_path / "gone.jpg")) is True


def test_a_photo_that_is_busy_for_a_moment_goes_on_a_later_try(photo, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "remove", busy(2, calls))

    assert delete_photo(str(photo), wait=0) is True

    assert len(calls) == 3
    assert not photo.exists()


def test_a_photo_that_stays_busy_is_given_up_and_reported_as_still_there(photo, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "remove", busy(100, calls))

    assert delete_photo(str(photo), wait=0) is False

    assert len(calls) == 3
    assert photo.exists()


def test_a_photo_that_cannot_be_deleted_does_not_crash_the_server(tmp_path):
    assert delete_photo(str(tmp_path), wait=0) is False  # a folder, so the delete fails
    assert tmp_path.exists()
