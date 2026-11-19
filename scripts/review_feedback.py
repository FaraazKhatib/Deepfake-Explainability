"""Review submitted verdict corrections locally. Run without arguments to list reports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.text.feedback import feedback_database, review_report


"""Review submitted verdict corrections locally. Run without arguments for an interactive review menu."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.text.feedback import feedback_database, review_report, get_all_feedback


def interactive_menu():
    items = get_all_feedback()
    if not items:
        print("\nNo feedback has been submitted yet.")
        return

    print("\n=======================================================")
    print("      VeriTrust AI — Collaborator Feedback Review      ")
    print("=======================================================\n")
    for idx, item in enumerate(items, 1):
        status = f"[APPROVED: {item['approved_verdict']}]" if item["is_approved"] else "[PENDING]"
        pub = f" ({item['publisher']})" if item['publisher'] else ""
        print(f"[{idx}] {status} {item['original_verdict']} -> {item['suggested_verdict']}{pub}")
        print(f"    Claims: {' | '.join(item['claims'][:2])}")
        if item.get("comment"):
            print(f"    Comment: {item['comment']}")
        if item["is_approved"]:
            print(f"    Reason: {item['review_reason']} ({item['review_source']})")
        print()

    choice = input("Enter item number to manage (or 'q' to quit): ").strip()
    if not choice or choice.lower() == "q":
        return

    try:
        idx = int(choice)
        if idx < 1 or idx > len(items):
            print("Invalid item number.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return

    selected = items[idx - 1]
    scan_id = selected["scan_id"]
    print(f"\nManaging Item [{idx}]: {selected['original_verdict']} -> {selected['suggested_verdict']}")
    print("Actions: (A)pprove / (R)evoke approval / (D)ismiss feedback / (C)ancel")
    action_in = input("Choose action [a/r/d/c]: ").strip().lower()

    if action_in == "a":
        source = input("Evidence URL (e.g. https://snopes.com/...): ").strip()
        while not source.startswith("http://") and not source.startswith("https://"):
            print("Please enter a valid HTTP/HTTPS URL.")
            source = input("Evidence URL: ").strip()
        reason = input("Review reason (why evidence justifies correction): ").strip()
        if not reason:
            reason = f"Verified correction to {selected['suggested_verdict']} via independent reporting."
        try:
            review_report(scan_id, "approve", source, reason)
            print("\n✓ Successfully APPROVED! Override will apply to future checks of this claim.")
        except Exception as exc:
            print(f"Error approving: {exc}")

    elif action_in == "r":
        try:
            review_report(scan_id, "revoke")
            print("\n✓ Successfully REVOKED approval. Default automated check restored.")
        except Exception as exc:
            print(f"Error revoking: {exc}")

    elif action_in == "d":
        confirm = input("Are you sure you want to dismiss and delete this feedback? [y/N]: ").strip().lower()
        if confirm == "y":
            try:
                review_report(scan_id, "dismiss")
                print("\n✓ Feedback item dismissed.")
            except Exception as exc:
                print(f"Error dismissing: {exc}")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "approve", "revoke", "dismiss", "interactive"], nargs="?", default=None)
    parser.add_argument("scan_id", nargs="?")
    parser.add_argument("--source", help="Evidence URL checked by the reviewer")
    parser.add_argument("--reason", help="Why the evidence justifies the correction")
    args = parser.parse_args()

    # Default to interactive menu if no action specified
    if args.action is None or args.action == "interactive":
        interactive_menu()
        return

    if args.action == "list":
        items = get_all_feedback()
        if not items:
            print("No feedback has been submitted.")
            return
        for item in items:
            status = f"[APPROVED: {item['approved_verdict']}]" if item["is_approved"] else "[PENDING]"
            pub = f" ({item['publisher']})" if item['publisher'] else ""
            print(f"{item['scan_id']}  {status}  {item['original_verdict']} -> {item['suggested_verdict']}{pub}")
            print("  " + "\n  ".join(item["claims"]))
            print(f"  User comment: {item['comment'] or '(none)'}\n  Active approval: {item['approved_verdict'] or '(none)'}\n")
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


if __name__ == "__main__":
    main()
