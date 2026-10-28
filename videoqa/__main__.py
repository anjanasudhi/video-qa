"""CLI: python -m videoqa extract <guideline.pdf> --brand Lumora [--backend mock --fixture rules.json]"""

import argparse

from .extract import extract_rules
from .llm import get_llm
from .store import save_brand


def main():
    parser = argparse.ArgumentParser(prog="videoqa")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="Extract video rules from a brand guideline PDF")
    ex.add_argument("pdf")
    ex.add_argument("--brand", required=True)
    ex.add_argument("--backend", default="gemini", choices=["gemini", "mock"])
    ex.add_argument("--fixture", help="JSON file returned by the mock backend")
    args = parser.parse_args()

    kwargs = {"fixture": args.fixture} if args.backend == "mock" else {}
    brand = extract_rules(args.pdf, args.brand, get_llm(args.backend, **kwargs))
    path = save_brand(brand)
    print(f"Saved {len(brand.rules)} rules for {brand.name} to {path}")
    for rule in brand.rules:
        print(f"  [{rule.severity.value:8}] {rule.id} ({rule.check_type.value})")


if __name__ == "__main__":
    main()
