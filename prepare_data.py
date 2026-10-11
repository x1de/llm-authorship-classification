"""Build a balanced, prompt-disjoint RAID subset for LLM authorship studies.

Input: RAID's labeled train_none.csv (or train.csv). No third-party package is
needed. The script never downloads data and never trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "llama-chat": "Llama",
    "mistral-chat": "Mistral",
    "cohere-chat": "Cohere",
}
DEFAULT_DOMAINS = (
    "abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"
)
REQUIRED_COLUMNS = {
    "id", "source_id", "model", "decoding", "repetition_penalty",
    "attack", "domain", "prompt", "generation",
}
OUTPUT_COLUMNS = (
    "LLM_name", "LLM_Input", "LLM_output", "source_model", "source_id",
    "record_id", "domain",
)


def normalized_text(value: str) -> str:
    # Compare text with uniform spacing while preserving the source text in CSVs.
    return re.sub(r"\s+", " ", value).strip()


def stable_key(seed: int, salt: str, value: str) -> str:
    payload = f"{seed}:{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path, domains: set[str]) -> tuple[dict, Counter, Counter]:
    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    counts: Counter = Counter()
    model_domain_counts: Counter = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"RAID CSV is missing columns: {', '.join(sorted(missing))}")

        # Use one generation setup so model labels are not tied to decoding settings.
        for row in reader:
            counts["rows_read"] += 1
            model = row["model"].strip()
            domain = row["domain"].strip()
            # Limit the study to the selected families and task domains.
            if model not in MODEL_LABELS:
                counts["other_model"] += 1
                continue
            if domain not in domains:
                counts["other_domain"] += 1
                continue
            if row["attack"].strip() != "none":
                counts["attacked"] += 1
                continue
            if row["decoding"].strip() != "sampling":
                counts["other_decoding"] += 1
                continue
            if row["repetition_penalty"].strip() not in ("no", ""):
                counts["repetition_penalty"] += 1
                continue
            # A paired example needs a source ID, prompt, and nonempty response.
            if not all(normalized_text(row[field] or "") for field in ("id", "source_id", "prompt", "generation")):
                counts["missing_essential_value"] += 1
                continue

            counts["candidate_rows"] += 1
            model_domain_counts[(domain, model)] += 1
            source_id = row["source_id"].strip()
            if model in groups[source_id]:
                # Multiple outputs for the same source and model are ambiguous.
                groups[source_id][model]["duplicate"] = "1"
                counts["repeated_source_model_rows"] += 1
            else:
                groups[source_id][model] = {
                    "id": row["id"].strip(),
                    "source_id": source_id,
                    "model": model,
                    "domain": domain,
                    "prompt": row["prompt"].strip(),
                    "generation": row["generation"].strip(),
                }
    return groups, counts, model_domain_counts


def eligible_groups(groups: dict, counts: Counter) -> dict[str, dict]:
    eligible = {}
    for source_id, by_model in groups.items():
        # Each retained prompt must have exactly one response from every family.
        if set(by_model) != set(MODEL_LABELS):
            counts["incomplete_model_groups"] += 1
            continue
        if any("duplicate" in record for record in by_model.values()):
            counts["ambiguous_model_groups"] += 1
            continue
        if len({record["domain"] for record in by_model.values()}) != 1:
            counts["mixed_domain_groups"] += 1
            continue
        if len({normalized_text(record["prompt"]) for record in by_model.values()}) != 1:
            counts["different_prompt_groups"] += 1
            continue
        eligible[source_id] = by_model

    # Keep one source group per prompt; two source IDs with the same prompt
    # must never land in different splits.
    prompt_owners = {}
    for source_id in sorted(eligible):
        prompt = normalized_text(next(iter(eligible[source_id].values()))["prompt"])
        if prompt in prompt_owners:
            counts["duplicate_prompt_groups"] += 1
            del eligible[source_id]
        else:
            prompt_owners[prompt] = source_id

    # Remove every group involved in an identical response, including a
    # response shared by two different model labels.
    output_owners: dict[str, set[str]] = defaultdict(set)
    for source_id, by_model in eligible.items():
        for record in by_model.values():
            output_owners[normalized_text(record["generation"])].add(source_id)
    duplicate_ids = set()
    for source_ids in output_owners.values():
        if len(source_ids) > 1:
            duplicate_ids.update(source_ids)
    for source_id, by_model in eligible.items():
        outputs = [normalized_text(record["generation"]) for record in by_model.values()]
        if len(set(outputs)) < len(outputs):
            duplicate_ids.add(source_id)
    for source_id in duplicate_ids:
        del eligible[source_id]
    counts["duplicate_output_groups"] = len(duplicate_ids)
    counts["eligible_groups"] = len(eligible)
    return eligible


def select_and_split(groups: dict, domains: tuple[str, ...], per_domain: int, seed: int) -> tuple[dict, dict]:
    n_train = int(per_domain * 0.70)
    n_validation = round(per_domain * 0.15)
    if min(n_train, n_validation, per_domain - n_train - n_validation) < 1:
        raise ValueError("--per-domain is too small to put prompts in every split")
    by_domain: dict[str, list[str]] = defaultdict(list)
    # Sample prompts within domains to avoid overrepresenting a task type.
    for source_id, by_model in groups.items():
        by_domain[next(iter(by_model.values()))["domain"]].append(source_id)
    available = {domain: len(by_domain[domain]) for domain in domains}
    short = {domain: count for domain, count in available.items() if count < per_domain}
    if short:
        raise ValueError(
            f"Not enough matched source prompts for {per_domain} per domain: {short}. "
            "Use --inspect-only to audit counts, then reduce --per-domain or --domains."
        )

    split_ids = {"train": [], "validation": [], "test": []}
    for domain in domains:
        # Hash ordering makes sampling and prompt-level splits repeatable.
        chosen = sorted(
            by_domain[domain], key=lambda source_id: stable_key(seed, "sample", source_id)
        )[:per_domain]
        chosen.sort(key=lambda source_id: stable_key(seed, "split", source_id))
        split_ids["train"].extend(chosen[:n_train])
        split_ids["validation"].extend(chosen[n_train:n_train + n_validation])
        split_ids["test"].extend(chosen[n_train + n_validation:])
    return split_ids, available


def write_outputs(output_dir: Path, split_ids: dict, groups: dict, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{split}.csv" for split in split_ids]
    paths.append(output_dir / "manifest.json")
    # Preserve earlier experiment inputs instead of silently replacing them.
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing outputs: {existing}")

    for split, source_ids in split_ids.items():
        with (output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for source_id in source_ids:
                # Keep all four model responses to a selected prompt in one split.
                for model, label in MODEL_LABELS.items():
                    record = groups[source_id][model]
                    writer.writerow({
                        "LLM_name": label,
                        "LLM_Input": record["prompt"],
                        "LLM_output": record["generation"],
                        "source_model": model,
                        "source_id": source_id,
                        "record_id": record["id"],
                        "domain": record["domain"],
                    })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to RAID train_none.csv or train.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/raid_v1"))
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--per-domain", type=int, default=250, help="Source prompts per domain; four rows per prompt")
    parser.add_argument("--seed", type=int, default=448)
    parser.add_argument("--inspect-only", action="store_true", help="Audit counts without writing a subset")
    args = parser.parse_args()
    if args.per_domain <= 0:
        parser.error("--per-domain must be positive")
    if len(set(args.domains)) != len(args.domains):
        parser.error("--domains must be unique")
    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")

    domains = tuple(args.domains)
    groups, counts, model_domain_counts = read_candidates(args.input, set(domains))
    groups = eligible_groups(groups, counts)
    available = Counter(next(iter(group.values()))["domain"] for group in groups.values())
    # Keep both the selection rules and rejection counts for later inspection.
    audit = {
        "source_file": args.input.name,
        "model_labels": MODEL_LABELS,
        "domains": domains,
        "filters": {"attack": "none", "decoding": "sampling", "repetition_penalty": "no"},
        "counts": dict(counts),
        "candidate_rows_by_domain_model": {
            domain: {model: model_domain_counts[(domain, model)] for model in MODEL_LABELS}
            for domain in domains
        },
        "eligible_matched_prompts_by_domain": {domain: available[domain] for domain in domains},
    }
    if args.inspect_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    split_ids, _ = select_and_split(groups, domains, args.per_domain, args.seed)
    # Hash the original file so the prepared sample has an identifiable source.
    manifest = {
        **audit,
        "source_sha256": file_sha256(args.input),
        "seed": args.seed,
        "prompts_per_domain": args.per_domain,
        "selection": "SHA-256 ordering of seed, salt, and source_id",
        "split_method": "70/15/15 by matched source_id within each domain",
        "split_prompt_counts": {split: len(ids) for split, ids in split_ids.items()},
        "split_row_counts": {split: len(ids) * len(MODEL_LABELS) for split, ids in split_ids.items()},
    }
    write_outputs(args.output_dir, split_ids, groups, manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["split_row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
