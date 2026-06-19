"""Download & prepare VersaPRM dataset: HF → local JSONL with domain mapping.

Source: UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled
Output: data/versaprm/versa_prm.jsonl
Domains: math / code / medical / general
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from datasets import load_dataset

CATEGORY_TO_DOMAIN: dict[str, str] = {
    "math": "math",
    "computer science": "code",
    "engineering": "code",
    "health": "medical",
    "biology": "medical",
    "psychology": "general",
    "other": "general",
    "law": "general",
    "economics": "general",
    "business": "general",
    "physics": "general",
    "chemistry": "general",
    "philosophy": "general",
    "history": "general",
}

OUT_PATH = Path("data/versaprm/versa_prm.jsonl")


def _parse_list(field: str | list | None) -> list:
    if field is None:
        return []
    if isinstance(field, list):
        return field
    return ast.literal_eval(field)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[versa] Loading HF dataset ...")
    ds = load_dataset(
        "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train"
    )
    print(f"[versa] {len(ds):,} rows loaded")

    written = 0
    skipped = 0
    domain_counts: dict[str, int] = {}

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in ds:
            try:
                steps = _parse_list(row["chain_of_thoughts"])
                labels = _parse_list(row["labels"])
            except (SyntaxError, ValueError):
                skipped += 1
                continue
            if not steps or len(steps) != len(labels):
                skipped += 1
                continue

            cat = (row["category"] or "").lower()
            domain = CATEGORY_TO_DOMAIN.get(cat, "general")

            sample = {
                "domain": domain,
                "category": cat,
                "question": row["question"],
                "steps": steps,
                "labels": labels,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"\n[versa] Wrote {written:,} rows ({skipped} skipped) → {OUT_PATH}")
    print(f"[versa] Size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    for d in ["math", "code", "medical", "general"]:
        n = domain_counts.get(d, 0)
        print(f"  {d:8s}  {n:>6,} samples")
    print("[versa] Done.")


if __name__ == "__main__":
    main()
