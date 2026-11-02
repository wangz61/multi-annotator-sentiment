import tkinter as tk

import pytest

import app
import db


@pytest.fixture(scope="module")
def tk_root():
    # One Tk per module: creating a second Tk after destroying the first is unreliable on Windows.
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def root(tk_root):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    yield window
    window.destroy()


def save_enabled(a):
    return not a.save_button.instate(["disabled"])


def test_save_needs_all_three_annotators(root, loaded):
    a = app.AnnotatorApp(root, loaded)
    assert a.current in {r.text for r in db.records(loaded)}
    assert not save_enabled(a)
    a.vars[0].set("Positive")
    a.vars[1].set("Negative")
    assert not save_enabled(a)
    assert "Annotator 3" in a.status_var.get()
    assert a.save() is None and db.count(loaded) == 1000
    a.vars[2].set("Sarcastic")
    assert save_enabled(a)


def test_saving_locks_the_record(root, loaded):
    a = app.AnnotatorApp(root, loaded)
    text = a.current
    for var, label in zip(a.vars, ("Positive", "Negative", "Positive")):
        var.set(label)
    assert a.save() == 1001
    assert db.records(loaded)[-1] == db.Record(1001, text, ("positive", "negative", "positive"), "ui")
    assert not save_enabled(a)
    assert all(box.instate(["disabled"]) for box in a.boxes)
    assert a.save() is None and db.count(loaded) == 1001
    assert "#1001" in a.status_var.get()
    a.generate()
    assert a.current != text
    assert all(var.get() == app.PLACEHOLDER for var in a.vars)
    assert not any(box.instate(["disabled"]) for box in a.boxes)


def test_stats_window(root, loaded, tmp_path):
    a = app.AnnotatorApp(root, loaded)
    a.open_stats()
    s = a.stats
    assert s.summary_vars["total"].get() == "1000"
    assert s.summary_vars["unique"].get() == "20"
    assert s.summary_vars["agreement"].get() == "86.00%"
    first = s.record_tree.item("1")["values"]
    assert first[5] == "tie" and first[8] == "33.33%"
    assert len(s.sentence_tree.get_children()) == 20
    assert "precision" in s.report.get("1.0", "end")

    for var in a.vars:
        var.set("Neutral")
    a.save()
    assert s.summary_vars["total"].get() == "1001"
    s.filter_var.set("New (this app)")
    s.fill_records()
    assert s.record_tree.get_children() == ("1001",)

    out = tmp_path / "out.csv"
    assert app.export_csv(out, s.records, s.result) == 1001
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("id,text,annotator_1")
    assert lines[-1].startswith("1001,") and lines[-1].endswith(",ui")
