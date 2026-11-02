import sqlite3

import pytest

import db
import import_data


def test_import_loads_every_row(conn):
    assert db.import_csv(conn, db.CSV_FILE) == (1000, 0)
    first = db.records(conn)[0]
    assert first == db.Record(1, "This movie was okay, not great.", ("sarcastic", "negative", "positive"), "import")


def test_import_is_idempotent(loaded):
    assert db.import_csv(loaded, db.CSV_FILE) == (0, 1000)
    assert db.count(loaded) == 1000


def test_bad_csv_imports_nothing(conn, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('id,text,annotations\n1,Fine.,"[""positive"",""neutral"",""neutral""]"\n'
                   '2,Odd.,"[""happy"",""neutral"",""neutral""]"\n', encoding="utf-8")
    with pytest.raises(db.DatasetError, match="line 3: unknown label 'happy'"):
        db.import_csv(conn, bad)
    assert db.count(conn) == 0


def test_wrong_annotation_count_is_rejected(conn, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text('id,text,annotations\n1,Fine.,"[""positive""]"\n', encoding="utf-8")
    with pytest.raises(db.DatasetError, match="expected 3 annotations"):
        db.import_csv(conn, bad)


def test_ensure_imported_only_fills_an_empty_database(conn):
    assert db.ensure_imported(conn) == 1000
    assert db.ensure_imported(conn) == 0


@pytest.mark.parametrize("sql", [
    "UPDATE annotations SET annotator_1 = 'neutral' WHERE id = 1",
    "DELETE FROM annotations WHERE id = 1",
    "DELETE FROM annotations",
])
def test_records_are_immutable(loaded, sql):
    with pytest.raises(sqlite3.IntegrityError, match="annotations are immutable"):
        loaded.execute(sql)
    assert db.records(loaded)[0].labels == ("sarcastic", "negative", "positive")
    assert db.count(loaded) == 1000


def test_replace_cannot_overwrite(loaded):
    with pytest.raises(sqlite3.IntegrityError):
        loaded.execute("INSERT OR REPLACE INTO annotations (id, text, annotator_1, annotator_2, annotator_3, source)"
                       " VALUES (1, 'x', 'neutral', 'neutral', 'neutral', 'ui')")
    assert db.records(loaded)[0].text == "This movie was okay, not great."


def test_add_annotation_appends_after_the_dataset(loaded):
    rec_id = db.add_annotation(loaded, "Meh.", ["neutral", "negative", "neutral"])
    assert rec_id == 1001
    assert db.records(loaded)[-1] == db.Record(1001, "Meh.", ("neutral", "negative", "neutral"), "ui")


@pytest.mark.parametrize("labels", [["neutral", "neutral"], ["neutral", "neutral", "happy"]])
def test_add_annotation_needs_three_valid_labels(loaded, labels):
    with pytest.raises(ValueError):
        db.add_annotation(loaded, "Meh.", labels)
    assert db.count(loaded) == 1000


def test_check_constraint_guards_raw_inserts(loaded):
    with pytest.raises(sqlite3.IntegrityError):
        loaded.execute("INSERT INTO annotations (text, annotator_1, annotator_2, annotator_3, source)"
                       " VALUES ('x', 'joy', 'neutral', 'neutral', 'ui')")


def test_random_sentence_comes_from_the_dataset_and_changes(loaded):
    texts = {r.text for r in db.records(loaded)}
    seen = set()
    current = None
    for _ in range(200):
        nxt = db.random_sentence(loaded, exclude=current)
        assert nxt in texts and nxt != current
        seen.add(nxt)
        current = nxt
    assert len(seen) == 20


def test_random_sentence_on_empty_db(conn):
    assert db.random_sentence(conn) is None


def test_import_script(tmp_path, capsys):
    target = tmp_path / "cli.db"
    assert import_data.main(["--db", str(target)]) == 0
    assert "Imported 1000 rows, skipped 0" in capsys.readouterr().out
    assert import_data.main(["--db", str(target)]) == 0
    assert "Imported 0 rows, skipped 1000" in capsys.readouterr().out
