"""CLI:
  python -m videoqa extract <guideline.pdf> --brand Lumora [--backend mock --fixture rules.json]
  python -m videoqa evaluate <video> --brand Lumora [--no-ai]
"""

import argparse

from .evaluate import evaluate
from .extract import extract_rules
from .llm import get_llm
from .store import load_brand, save_brand
from .vision import GeminiVideoReviewer


def cmd_extract(args):
    kwargs = {"fixture": args.fixture} if args.backend == "mock" else {}
    brand = extract_rules(args.pdf, args.brand, get_llm(args.backend, **kwargs))
    path = save_brand(brand)
    print(f"Saved {len(brand.rules)} rules for {brand.name} to {path}")
    for rule in brand.rules:
        print(f"  [{rule.severity.value:8}] {rule.id} ({rule.check_type.value})")


def cmd_evaluate(args):
    brand = load_brand(args.brand)
    reviewer = None if args.no_ai else GeminiVideoReviewer()
    report = evaluate(args.video, brand, reviewer)
    print(f"{report.video}: score {report.score}/100 for {report.brand}")
    for r in report.results:
        when = f" @ {', '.join(f'{t:g}s' for t in r.timestamps)}" if r.timestamps else ""
        print(f"  {r.status.value.upper():7} [{r.severity.value}] {r.title}: {r.detail}{when}")


def main():
    parser = argparse.ArgumentParser(prog="videoqa")
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="Extract video rules from a brand guideline PDF")
    ex.add_argument("pdf")
    ex.add_argument("--brand", required=True)
    ex.add_argument("--backend", default="gemini", choices=["gemini", "mock"])
    ex.add_argument("--fixture", help="JSON file returned by the mock backend")
    ex.set_defaults(func=cmd_extract)

    ev = sub.add_parser("evaluate", help="Check a video against a brand's rules")
    ev.add_argument("video")
    ev.add_argument("--brand", required=True)
    ev.add_argument("--no-ai", action="store_true", help="Run only the technical checks")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
