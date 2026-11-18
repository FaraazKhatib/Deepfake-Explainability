"""Review submitted verdict corrections locally. Run without arguments to list reports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.text.feedback import feedback_database, review_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "approve", "revoke"], nargs="?", default="list")
    parser.add_argument("scan_id", nargs="?")
    parser.add_argument("--source", help="Evidence URL checked by the reviewer")
    parser.add_argument("--reason", help="Why the evidence justifies the correction")
    args = parser.parse_args()
    if args.action == "list":
        with feedback_database() as db:
            rows = db.execute("""
                SELECT f.scan_id,claims_json,publisher,original_verdict,suggested_verdict,comment,r.verdict
                FROM feedback f LEFT JOIN reviewed_verdicts r ON r.report_id=f.scan_id
                ORDER BY f.updated_at DESC
            """).fetchall()
        if not rows:
            print("No feedback has been submitted.")
        for scan, claims, publisher, original, suggested, comment, approved in rows:
            print(f"{scan}  {publisher}  {original} -> {suggested}")
            print("  " + "\n  ".join(json.loads(claims)))
            print(f"  User comment: {comment or '(none)'}\n  Active approval: {approved or '(none)'}")
        return
    if not args.scan_id:
        parser.error("Supply the scan ID shown by list.")
    source, reason = args.source or "", args.reason or ""
    if args.action == "approve":
        source = source or input("Evidence URL you checked: ")
        reason = reason or input("Why does it support this correction? ")
    try:
        review_report(args.scan_id, args.action, source, reason)
    except ValueError as exc:
        parser.error(str(exc))
    print("Review saved. Start a new scan to see the current verdict.")


if __name__ == "__main__":
    main()
