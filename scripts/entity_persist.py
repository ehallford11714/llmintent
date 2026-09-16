"""Write locked entity slots for a prompt (GPT-2 by default; pass qwen:27b)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Persist digit/noun slots from a forward")
    p.add_argument("text", nargs="?", default="8 minus 2. The answer is")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args(argv)

    from llmintent.entity import persist_entities
    from llmintent.suite import load_suite_model, resolve_model_spec

    spec = resolve_model_spec(model=args.model, use_env=False)
    four = True if args.load_in_4bit else None
    if four is None and spec is not None and getattr(spec, "size", None) == "27b":
        four = True
    bundle = load_suite_model(model=args.model, load_in_4bit=four)
    report = persist_entities(bundle, args.text)
    payload = report.to_dict()
    payload["model"] = getattr(bundle, "name", args.model)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
