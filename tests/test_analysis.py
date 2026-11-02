import pytest

import analysis
import db


@pytest.mark.parametrize("labels, expected", [
    (("positive", "positive", "positive"), ("positive", 1.0)),
    (("sarcastic", "negative", "sarcastic"), ("sarcastic", 2 / 3)),
    (("sarcastic", "negative", "positive"), ("tie", 1 / 3)),
])
def test_resolve(labels, expected):
    label, agreement = analysis.resolve(labels)
    assert label == expected[0]
    assert agreement == pytest.approx(expected[1])


def test_summary_of_the_dataset(loaded):
    s = analysis.summarize(db.records(loaded))
    assert (s.total, s.unique_texts, s.ties) == (1000, 20, 105)
    assert s.average_agreement == pytest.approx(0.86)


def test_by_sentence(loaded):
    rows = analysis.by_sentence(db.records(loaded))
    assert len(rows) == 20
    assert sum(r.records for r in rows) == 1000
    oh_sure = next(r for r in rows if r.text == "Oh sure, best idea ever.")
    assert oh_sure.majority == "sarcastic"
    assert sum(oh_sure.votes.values()) == oh_sure.records * 3


def test_classifier_on_the_dataset(loaded):
    records = db.records(loaded)
    result = analysis.train_classifier(records, seed=42)
    assert result.ok
    assert result.excluded_ties == 105
    assert result.train_size + result.test_size == 895
    assert 0.6 < result.accuracy <= result.ceiling
    assert result.unseen_accuracy is not None and result.unseen_accuracy < result.accuracy
    assert set(result.predictions) == {r.id for r in records}
    assert all(0 < conf <= 1 and label in db.LABELS for label, conf in result.predictions.values())
    for label in db.LABELS:
        assert label in result.report
    text = analysis.describe(result)
    assert "precision" in text and "GroupKFold" in text


@pytest.mark.parametrize("seed", range(10))
def test_accuracy_never_beats_the_ceiling(loaded, seed):
    result = analysis.train_classifier(db.records(loaded), seed=seed)
    assert result.accuracy <= result.ceiling


def test_label_ceiling():
    assert analysis.label_ceiling(["a", "a", "a", "b"], ["x", "x", "y", "z"]) == 0.75


def test_classifier_is_deterministic_per_seed(loaded):
    records = db.records(loaded)
    a = analysis.train_classifier(records, seed=7)
    b = analysis.train_classifier(records, seed=7)
    assert a.accuracy == b.accuracy and a.report == b.report


def test_rerun_includes_new_annotations(loaded):
    before = analysis.train_classifier(db.records(loaded))
    for _ in range(5):
        db.add_annotation(loaded, "Meh.", ["neutral", "neutral", "neutral"])
    after = analysis.train_classifier(db.records(loaded))
    assert after.train_size + after.test_size == before.train_size + before.test_size + 5
    assert 1005 in after.predictions


def test_classifier_needs_enough_data(conn):
    db.add_annotation(conn, "Great job!", ["positive"] * 3)
    result = analysis.train_classifier(db.records(conn))
    assert not result.ok
    assert "Need at least 10" in analysis.describe(result)


def test_classifier_skips_ties(conn):
    for _ in range(12):
        db.add_annotation(conn, "Meh.", ["neutral", "positive", "negative"])
    result = analysis.train_classifier(db.records(conn))
    assert not result.ok and result.excluded_ties == 12
