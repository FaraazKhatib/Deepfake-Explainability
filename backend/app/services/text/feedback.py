"""Store user-reported corrections separately from detector verdicts."""
from contextlib import closing, contextmanager
from urllib.parse import urlsplit
import logging
import json
import os
from pathlib import Path
import sqlite3
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class VerdictFeedback(BaseModel):
    scan_id: UUID
    original_verdict: str = Field(min_length=1, max_length=100)
    suggested_verdict: Literal["supported", "contradicted", "disputed", "unverified"]
    claims: list[Annotated[str, Field(min_length=1, max_length=5000)]] = Field(min_length=1, max_length=3)
    publisher: str = Field(default="", max_length=255)
    comment: str = Field(default="", max_length=1000)


def database_path() -> Path:
    default = Path(__file__).resolve().parents[4] / "results" / "fact_check_feedback.sqlite3"
    return Path(os.environ.get("VT_FEEDBACK_DB") or default)


@contextmanager
def feedback_database():
    database = database_path()
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                scan_id TEXT PRIMARY KEY, original_verdict TEXT NOT NULL, claims_json TEXT NOT NULL,
                publisher TEXT NOT NULL, suggested_verdict TEXT NOT NULL, comment TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reviewed_verdicts (
                claim_key TEXT PRIMARY KEY, report_id TEXT NOT NULL, verdict TEXT NOT NULL,
                reason TEXT NOT NULL, source TEXT NOT NULL,
                reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        yield connection


def save_feedback(report: VerdictFeedback) -> None:
    with feedback_database() as connection:
        connection.execute("""
            INSERT INTO feedback (scan_id, original_verdict, claims_json, publisher, suggested_verdict, comment)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET suggested_verdict=excluded.suggested_verdict,
                comment=excluded.comment, updated_at=CURRENT_TIMESTAMP
        """, (str(report.scan_id), report.original_verdict, json.dumps(report.claims, ensure_ascii=False),
              report.publisher, report.suggested_verdict, report.comment))

def claim_key(claims):
    # Only whitespace and claim order are immaterial. Keep names, punctuation,
    # numbers, negation and case intact.
    return json.dumps(sorted(" ".join(c.split()) for c in claims), ensure_ascii=False)


def review_report(scan_id, action, source="", reason=""):
    if action not in {"approve", "revoke"}:
        raise ValueError("Choose approve or revoke.")
    with feedback_database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT claims_json, suggested_verdict FROM feedback WHERE scan_id=?", (scan_id,)).fetchone()
        if not row:
            raise ValueError("Feedback report not found.")
        key = claim_key(json.loads(row[0]))
        if action == "revoke":
            deleted = db.execute("DELETE FROM reviewed_verdicts WHERE claim_key=? AND report_id=?", (key, scan_id))
            if not deleted.rowcount:
                raise ValueError("This report has no active approval.")
            return
        parsed = urlsplit(source.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not reason.strip():
            raise ValueError("Approval requires an HTTP(S) evidence link and a review reason.")
        db.execute("""
            INSERT OR REPLACE INTO reviewed_verdicts (claim_key,report_id,verdict,source,reason)
            VALUES (?,?,?,?,?)
        """, (key, scan_id, row[1], source.strip(), reason.strip()))


def apply_review(result):
    path = database_path()
    if not path.is_file():
        return result
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            # Databases created before review support contain only feedback.
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='reviewed_verdicts'").fetchone():
                return result
            row = db.execute("SELECT verdict,reason,source,reviewed_at FROM reviewed_verdicts WHERE claim_key=?",
                             (claim_key(c["claim"] for c in result["claims"]),)).fetchone()
    except sqlite3.Error:
        logging.getLogger(__name__).warning("Review database unavailable", exc_info=True)
        return result
    if row:
        verdict, reason, source, reviewed_at = row
        result["review"] = {"automated_verdict": result["verdict"], "reason": reason,
                            "source": source, "reviewed_at": reviewed_at}
        result["verdict"] = verdict.capitalize()
        result["summary"] = f"Reviewed verdict: {verdict.capitalize()}. {reason}"
        result["rationale"] = reason
        result["explanation"] = f'Automated verdict: {result["review"]["automated_verdict"]}. The claim details below are from the automated check.'
    return result
