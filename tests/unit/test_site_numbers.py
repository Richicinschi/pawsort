import pytest

from orchestrator import db, numbers


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def tagged(conn, site="A12", day="2026-11-10", marked=False, label="bobcat"):
    """A done job that came in on the given day."""
    job_id = db.create_job(conn, "cat.jpg", site)
    db.claim_job(conn)
    db.set_result(conn, job_id, label, 0.6 if marked else 0.9, marked)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (f"{day}T12:00:00+00:00", job_id))
    return job_id


def row(conn, site="A12", week="2026-W46"):
    return next(r for r in numbers.site_numbers(conn) if (r["site_code"], r["week"]) == (site, week))


@pytest.mark.parametrize("day,week", [("2026-11-09", "2026-W46"), ("2026-11-15", "2026-W46"),
                                      ("2026-11-16", "2026-W47"), ("2027-01-01", "2026-W53"),
                                      ("2027-01-04", "2027-W01")])
def test_week_runs_from_monday_to_sunday(day, week):
    assert numbers.week_of(f"{day}T12:00:00+00:00") == week


def test_tags_are_counted_per_camera_site_and_week_of_the_upload(conn):
    for site, day in [("A12", "2026-11-10"), ("A12", "2026-11-11"), ("B7", "2026-11-10"), ("A12", "2026-11-17")]:
        tagged(conn, site, day)

    counts = {(r["site_code"], r["week"]): r["tags"] for r in numbers.site_numbers(conn)}

    assert counts == {("A12", "2026-W46"): 2, ("B7", "2026-W46"): 1, ("A12", "2026-W47"): 1}


def test_newest_week_comes_first_and_sites_in_order(conn):
    for site, day in [("B7", "2026-11-10"), ("A12", "2026-11-10"), ("A12", "2026-11-17")]:
        tagged(conn, site, day)
    assert [(r["week"], r["site_code"]) for r in numbers.site_numbers(conn)] == [
        ("2026-W47", "A12"), ("2026-W46", "A12"), ("2026-W46", "B7")]


def test_only_a_tag_that_was_given_counts(conn):
    tagged(conn)
    db.create_job(conn, "waiting.jpg", "A12")
    failed = db.create_job(conn, "broken.jpg", "A12")
    db.claim_job(conn)
    db.claim_job(conn)
    db.set_failed(conn, failed, "broken photo")
    conn.execute("UPDATE jobs SET created_at = '2026-11-10T12:00:00+00:00'")

    assert row(conn)["tags"] == 1


def test_share_of_tags_that_needed_review(conn):
    for marked in (True, False, False, False):
        tagged(conn, marked=marked)
    assert row(conn)["needs_review"] == 0.25


def test_tag_a_person_checked_still_counts_as_one_that_needed_review(conn):
    job_id = tagged(conn, marked=True)
    db.save_review(conn, job_id, "bobcat")
    assert row(conn)["needs_review"] == 1.0


def test_share_of_the_checked_tags_that_were_corrected(conn):
    confirmed, corrected, approved = (tagged(conn, marked=True) for _ in range(3))
    tagged(conn, marked=True)
    db.save_review(conn, confirmed, "bobcat")
    db.save_review(conn, corrected, "coyote")
    db.save_review(conn, approved, "deer")
    db.decide_correction(conn, approved, "approved")

    found = row(conn)

    assert (found["checked"], found["corrected"]) == (3, round(2 / 3, 4))


def test_rejected_correction_does_not_count_as_corrected(conn):
    job_id, other = tagged(conn, marked=True), tagged(conn, marked=True)
    db.save_review(conn, job_id, "coyote")
    db.save_review(conn, other, "deer")
    db.decide_correction(conn, job_id, "rejected")

    assert row(conn)["corrected"] == 0.5


def test_site_where_nobody_checked_a_tag_has_no_corrected_share(conn):
    tagged(conn, marked=True)
    found = row(conn)
    assert (found["checked"], found["corrected"]) == (0, None)


@pytest.mark.parametrize("marked,warning", [(2, False), (3, True)])
def test_warning_only_when_more_than_40_percent_needed_review(conn, marked, warning):
    for number in range(5):
        tagged(conn, marked=number < marked)
    assert row(conn)["warning"] is warning


def test_the_limit_for_a_warning_is_40_percent():
    assert numbers.WARN_SHARE == 0.40


def test_warning_belongs_to_one_site_and_one_week(conn):
    tagged(conn, "A12", "2026-11-10", marked=True)
    tagged(conn, "B7", "2026-11-10")
    tagged(conn, "A12", "2026-11-17")

    assert [(r["site_code"], r["week"]) for r in numbers.site_numbers(conn) if r["warning"]] == [("A12", "2026-W46")]


def test_job_without_a_site_code_is_counted_under_no_site(conn):
    tagged(conn, site=None)
    assert row(conn, site=None)["tags"] == 1


def test_site_is_new_when_its_first_photo_came_after_the_training(conn):
    tagged(conn, "OLD1", "2026-11-01")
    tagged(conn, "OLD1", "2026-11-20")
    tagged(conn, "NEW2", "2026-11-12")
    tagged(conn, "NEW1", "2026-11-11")
    tagged(conn, "SAME", "2026-11-10")

    assert numbers.new_sites(conn, "2026-11-10") == ["NEW1", "NEW2"]


def test_photo_that_is_still_waiting_already_makes_a_site_known(conn):
    db.create_job(conn, "waiting.jpg", "C3")
    conn.execute("UPDATE jobs SET created_at = '2026-11-12T12:00:00+00:00'")
    assert numbers.new_sites(conn, "2026-11-10") == ["C3"]


def test_without_a_training_date_no_site_counts_as_new(conn):
    tagged(conn, "NEW1", "2026-11-11")
    assert numbers.new_sites(conn, None) == []
