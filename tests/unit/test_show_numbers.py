from interface.app import sample_note, show_numbers, show_sample_job, site_warnings, training_reasons


def row(site="B7", week="2026-W46", tags=20, needs_review=0.45, checked=4, corrected=0.25):
    return {"site_code": site, "week": week, "tags": tags, "needs_review": needs_review, "checked": checked,
            "corrected": corrected, "warning": needs_review > 0.4}


def numbers(*rows, for_training=0, plan_training=False, new_sites=(), trained_on="2026-11-06"):
    return {"rows": list(rows), "warn_share": 0.4, "for_training": for_training, "plan_training": plan_training,
            "new_sites": list(new_sites), "trained_on": trained_on}


def test_row_shows_the_shares_as_percentages():
    assert show_numbers(row()) == {"site_code": "B7", "week": "2026-W46", "tags": 20, "needs_review": "45%",
                                   "corrected": "25% (of 4 checked)", "warning": True}


def test_row_of_a_site_where_nobody_checked_a_tag_says_so():
    assert show_numbers(row(checked=0, corrected=None))["corrected"] == "no tag checked yet"


def test_row_without_a_site_code():
    assert show_numbers(row(site=None))["site_code"] == "unknown"


def test_warning_names_the_site_the_week_the_share_and_the_40_percent():
    assert site_warnings(numbers(row(), row("A12", needs_review=0.4))) == [
        "Camera site B7, week 2026-W46: 45% of the tags needed review. That is more than 40%."]


def test_no_reason_to_plan_a_training_round():
    assert training_reasons(numbers(row(needs_review=0.1), for_training=499)) == []


def test_enough_approved_corrections_are_a_reason():
    assert training_reasons(numbers(for_training=500, plan_training=True)) == [
        "500 approved corrections wait for the next training."]


def test_a_new_camera_site_is_a_reason():
    assert training_reasons(numbers(new_sites=["C3", "D4"])) == [
        "Camera site C3 is new: its first photo came after the current model was trained on 2026-11-06.",
        "Camera site D4 is new: its first photo came after the current model was trained on 2026-11-06."]


def test_a_warning_is_a_reason():
    assert training_reasons(numbers(row(), row("A12", needs_review=0.2))) == [
        "Camera site B7 had more than 40% needs review in week 2026-W46."]


def test_every_reason_that_holds_is_listed():
    reasons = training_reasons(numbers(row(), for_training=600, plan_training=True, new_sites=["C3"]))
    assert len(reasons) == 3 and reasons[0].startswith("600 approved corrections")


def test_sample_note_counts_the_picked_the_checked_and_the_changed():
    assert sample_note({"week": "2026-W46", "picked": 50, "checked": 3, "changed": 1}) == (
        "Sample of week 2026-W46: 50 sure tags picked, 3 of them checked, 1 of those changed.")


def test_sample_note_of_a_week_without_sure_tags():
    assert sample_note({"week": "2026-W46", "picked": 0, "checked": 0, "changed": 0}) == (
        "Sample of week 2026-W46: no sure tag came in since the sample before, so there is nothing to check.")


def test_sample_row_keeps_the_real_label_for_the_answer():
    job = {"id": "job-1", "site_code": None, "label": "mountain_lion", "probability": 0.9349, "model_version": None}
    assert show_sample_job(job) == {"job_id": "job-1", "site_code": "unknown", "label": "mountain_lion",
                                    "label_text": "mountain lion", "percent": 93, "model_version": "unknown"}


def test_share_just_above_the_limit_is_never_shown_as_the_limit_itself():
    close = row(tags=42, needs_review=round(17 / 42, 4))
    assert show_numbers(close)["needs_review"] == "40.48%"
    assert site_warnings(numbers(close)) == [
        "Camera site B7, week 2026-W46: 40.48% of the tags needed review. That is more than 40%."]
