"""Import multi_annotator_dataset_list.csv into annotations.db.

    python import_data.py [--csv PATH] [--db PATH]
"""

import argparse
import sys
from pathlib import Path

import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=db.CSV_FILE)
    parser.add_argument("--db", type=Path, default=db.DB_FILE)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    conn = db.connect(args.db)
    try:
        inserted, skipped = db.import_csv(conn, args.csv)
    except db.DatasetError as err:
        print(f"Import failed, nothing written: {err}", file=sys.stderr)
        return 1
    finally:
        total = db.count(conn)
        conn.close()
    print(f"Imported {inserted} rows, skipped {skipped} already present. {args.db.name} now holds {total} records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
