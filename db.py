"""SQLite storage for annotation records. Saved records can never be changed."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "annotations.db"
CSV_FILE = HERE / "multi_annotator_dataset_list.csv"

LABELS = ("positive", "neutral", "negative", "sarcastic")
ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3")

_LABEL_CHECK = "IN ('positive','neutral','negative','sarcastic')"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS annotations (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL CHECK (length(trim(text)) > 0),
    annotator_1 TEXT NOT NULL CHECK (annotator_1 {_LABEL_CHECK}),
    annotator_2 TEXT NOT NULL CHECK (annotator_2 {_LABEL_CHECK}),
    annotator_3 TEXT NOT NULL CHECK (annotator_3 {_LABEL_CHECK}),
    source      TEXT NOT NULL CHECK (source IN ('import','ui')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS annotations_text ON annotations(text);
CREATE TRIGGER IF NOT EXISTS annotations_no_update BEFORE UPDATE ON annotations
BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS annotations_no_delete BEFORE DELETE ON annotations
BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
-- INSERT OR REPLACE removes the old row without firing delete triggers, so reject reused ids up front.
CREATE TRIGGER IF NOT EXISTS annotations_no_replace BEFORE INSERT ON annotations
WHEN EXISTS (SELECT 1 FROM annotations WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
"""


class DatasetError(ValueError):
    """A CSV row failed validation; nothing was imported."""


@dataclass(frozen=True)
class Record:
    id: int
    text: str
    labels: tuple[str, str, str]
    source: str


def connect(path: Path | str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _parse_row(line: int, row: dict[str, str]) -> tuple[int, str, list[str]]:
    try:
        rec_id = int(row["id"])
        text = row["text"].strip()
        labels = json.loads(row["annotations"])
    except (KeyError, TypeError, ValueError) as err:
        raise DatasetError(f"line {line}: cannot read row ({err})") from err
    if not text:
        raise DatasetError(f"line {line}: empty text")
    if not isinstance(labels, list) or len(labels) != 3:
        raise DatasetError(f"line {line}: expected 3 annotations, got {labels!r}")
    labels = [str(label).strip().lower() for label in labels]
    bad = [label for label in labels if label not in LABELS]
    if bad:
        raise DatasetError(f"line {line}: unknown label {bad[0]!r}")
    return rec_id, text, labels


def import_csv(conn: sqlite3.Connection, csv_path: Path | str = CSV_FILE) -> tuple[int, int]:
    """Import the dataset. Returns (inserted, skipped); rows whose id exists are skipped.

    Every row is validated before anything is written, so a bad file imports nothing.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = [_parse_row(i, row) for i, row in enumerate(csv.DictReader(handle), start=2)]
    existing = {row[0] for row in conn.execute("SELECT id FROM annotations")}
    new = [(rec_id, text, *labels) for rec_id, text, labels in rows if rec_id not in existing]
    with conn:
        conn.executemany(
            "INSERT INTO annotations (id, text, annotator_1, annotator_2, annotator_3, source)"
            " VALUES (?, ?, ?, ?, ?, 'import')",
            new,
        )
    return len(new), len(rows) - len(new)


def ensure_imported(conn: sqlite3.Connection, csv_path: Path | str = CSV_FILE) -> int:
    """Import the dataset into an empty database. Returns the number of rows imported."""
    if count(conn) or not Path(csv_path).exists():
        return 0
    return import_csv(conn, csv_path)[0]


def add_annotation(conn: sqlite3.Connection, text: str, labels: tuple[str, ...] | list[str]) -> int:
    """Save a new record from the annotator UI and return its id."""
    labels = [label.lower() for label in labels]
    if len(labels) != 3 or any(label not in LABELS for label in labels):
        raise ValueError("every annotator needs one of: " + ", ".join(LABELS))
    with conn:
        cur = conn.execute(
            "INSERT INTO annotations (text, annotator_1, annotator_2, annotator_3, source)"
            " VALUES (?, ?, ?, ?, 'ui')",
            (text, *labels),
        )
    return int(cur.lastrowid)


def records(conn: sqlite3.Connection) -> list[Record]:
    rows = conn.execute(
        "SELECT id, text, annotator_1, annotator_2, annotator_3, source FROM annotations ORDER BY id"
    )
    return [Record(r[0], r[1], (r[2], r[3], r[4]), r[5]) for r in rows]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]


def random_sentence(conn: sqlite3.Connection, exclude: str | None = None) -> str | None:
    """A random distinct sentence from the database, different from `exclude` when possible."""
    row = conn.execute(
        "SELECT text FROM (SELECT DISTINCT text FROM annotations) WHERE text IS NOT ?"
        " ORDER BY random() LIMIT 1",
        (exclude,),
    ).fetchone()
    if row is None and exclude is not None:
        row = conn.execute("SELECT text FROM annotations LIMIT 1").fetchone()
    return row[0] if row else None
