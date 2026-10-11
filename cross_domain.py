"""Compare in-domain and unseen-domain output-only classification."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import (
    BiLSTM, TextCNN, TextDataset, build_vocabulary, collate_batch, evaluate,
    read_rows, set_seed, train_one,
)


def original_model_score(architecture, test_rows, vocabulary, config, run_dir, device):
    # Score the original model on the same domain-specific rows as the new model.
    if architecture == "cnn":
        model = TextCNN(len(vocabulary), config["embedding_dim"], config["cnn_channels"], config["dropout"])
    else:
        model = BiLSTM(len(vocabulary), config["embedding_dim"], config["lstm_hidden"], config["dropout"])
    model.load_state_dict(torch.load(run_dir / f"{architecture}_output" / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    dataset = TextDataset(test_rows, vocabulary, "output", config["prompt_limit"], config["output_limit"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_batch)
    return evaluate(model, loader, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/raid_v1"))
    parser.add_argument("--original-runs", type=Path, default=Path("runs/raid_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cross_domain"))
    parser.add_argument("--holdout-domains", nargs="+", default=["news", "poetry"])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to replace existing runs: {args.output_dir}")

    # Reuse the original settings so the domain exclusion is the main change.
    original_config = json.loads((args.original_runs / "configuration.json").read_text(encoding="utf-8"))
    original_vocabulary = json.loads((args.original_runs / "vocabulary.json").read_text(encoding="utf-8"))
    all_rows = {split: read_rows(args.data_dir / f"{split}.csv") for split in ("train", "validation", "test")}
    known_domains = {row["domain"] for rows in all_rows.values() for row in rows}
    if len(set(args.holdout_domains)) != len(args.holdout_domains) or not set(args.holdout_domains) <= known_domains:
        parser.error("Holdout domains must be unique domains present in the prepared data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {"holdout_domains": args.holdout_domains, "protocol": "Train and validate on seven domains; test on the original test rows of the held-out domain.", "results": {}}
    args.output_dir.mkdir(parents=True)
    for domain in args.holdout_domains:
        # Remove this domain from both fitting splits, but retain its test rows.
        rows = {
            "train": [row for row in all_rows["train"] if row["domain"] != domain],
            "validation": [row for row in all_rows["validation"] if row["domain"] != domain],
            "test": [row for row in all_rows["test"] if row["domain"] == domain],
        }
        # Prompt IDs must stay disjoint even after filtering by domain.
        assert not ({r["source_id"] for r in rows["train"]} & {r["source_id"] for r in rows["test"]})
        # Refit token IDs after exclusion so held-out text cannot shape the vocabulary.
        vocabulary = build_vocabulary(rows["train"], original_config["max_vocab"], original_config["min_frequency"])
        domain_dir = args.output_dir / f"heldout_{domain}"
        domain_dir.mkdir()
        (domain_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["results"][domain] = {"train_rows": len(rows["train"]), "validation_rows": len(rows["validation"]), "test_rows": len(rows["test"]), "vocabulary_size": len(vocabulary)}
        # Match the original training budget and model dimensions.
        train_args = argparse.Namespace(**{key: original_config[key] for key in (
            "seed", "epochs", "patience", "batch_size", "learning_rate", "weight_decay",
            "embedding_dim", "cnn_channels", "lstm_hidden", "dropout", "prompt_limit", "output_limit",
        )})
        train_args.output_dir = domain_dir
        for architecture in ("cnn", "lstm"):
            set_seed(train_args.seed)
            # Score both models on the identical held-out domain test rows.
            unseen = train_one(architecture, "output", rows, vocabulary, train_args, device)
            original = original_model_score(architecture, rows["test"], original_vocabulary, original_config, args.original_runs, device)
            summary["results"][domain][architecture] = {
                "original_in_domain_test": original,
                "unseen_domain_test": unseen["test"],
                "best_epoch": unseen["best_epoch"],
                "best_validation_macro_f1": unseen["best_validation_macro_f1"],
            }
            # Save each completed comparison in case a later run is interrupted.
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(domain, architecture, "original F1", round(original["macro_f1"], 4), "unseen F1", round(unseen["test"]["macro_f1"], 4), flush=True)
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
