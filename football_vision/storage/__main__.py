"""Import video-analysis JSON and query local SQLite reports."""

import argparse
import json
from pathlib import Path
import sqlite3

from .database import REPORTS, import_analysis, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("analysis", type=Path)
    ingest.add_argument("--db", type=Path, required=True)
    query = commands.add_parser("report")
    query.add_argument("name", choices=REPORTS)
    query.add_argument("--db", type=Path, required=True)
    query.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "import":
            run_id, inserted = import_analysis(args.db, json.loads(args.analysis.read_text()))
            result = {"run_id": run_id, "inserted": inserted}
        else:
            result = report(args.db, args.run_id, args.name)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        parser.exit(2, f"Storage failed: {exc}\n")


if __name__ == "__main__":
    main()
