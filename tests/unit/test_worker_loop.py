import json
import os

import httpx

from ai_server.worker import poll

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fake_server(*answers, photo="bobcat_1.jpg"):
    """A middle server that only exists in this test. Each answer is a status code for one claim or report."""
    left = list(answers)
    seen = []

    def answer(request):
        seen.append(request.url.path)
        if request.url.path == "/jobs/job-1/photo":
            with open(os.path.join(DATA, photo), "rb") as f:
                return httpx.Response(200, content=f.read())
        status = left.pop(0)
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status, json={"id": "job-1"})

    return httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer)), seen


def stand_in(image):
    return {"bobcat": 0.9, "coyote": 0.1}


def test_empty_queue_waits_a_moment(work):
    client, _ = fake_server(204)
    waits = []
    poll(client, stand_in, work, wait=waits.append)
    assert waits == [0.5]


def test_server_error_when_asking_for_a_job_only_makes_it_wait(work):
    client, _ = fake_server(500)
    waits = []
    poll(client, stand_in, work, wait=waits.append)
    assert waits == [1.0]


def test_server_error_when_reporting_only_makes_it_wait(work):
    client, seen = fake_server(200, 500)
    waits = []

    poll(client, stand_in, work, wait=waits.append)

    assert seen == ["/jobs/claim", "/jobs/job-1/photo", "/jobs/job-1/result"]
    assert waits == [1.0]
    assert os.listdir(work) == []


def test_refused_late_answer_only_makes_it_wait(work):
    client, _ = fake_server(200, 409)
    waits = []
    poll(client, stand_in, work, wait=waits.append)
    assert waits == [1.0]


def test_no_connection_only_makes_it_wait(work):
    client, _ = fake_server(httpx.ConnectError("refused"))
    waits = []
    poll(client, stand_in, work, wait=waits.append)
    assert waits == [1.0]


def test_loop_carries_on_with_the_next_job_after_a_server_error(work):
    client, seen = fake_server(503, 200, 200, 204)
    waits = []

    for _ in range(3):
        poll(client, stand_in, work, wait=waits.append)

    assert seen == ["/jobs/claim", "/jobs/claim", "/jobs/job-1/photo", "/jobs/job-1/result", "/jobs/claim"]
    assert waits == [1.0, 0.5]


def test_broken_photo_is_still_reported(work):
    reports = []

    def answer(request):
        if request.url.path == "/jobs/claim":
            return httpx.Response(200, json={"id": "job-1"})
        if request.url.path == "/jobs/job-1/photo":
            with open(os.path.join(DATA, "not_an_image.jpg"), "rb") as f:
                return httpx.Response(200, content=f.read())
        reports.append(json.loads(request.content))
        return httpx.Response(200, json={})

    client = httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer))
    poll(client, stand_in, work, wait=lambda seconds: None)
    assert "error" in reports[0]
