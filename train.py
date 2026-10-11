"""Train and evaluate TextCNN and bidirectional LSTM on prepared RAID data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


LABELS = ("ChatGPT", "Llama", "Mistral", "Cohere")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]|\n", re.UNICODE)
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"


def tokenize(text: str) -> list[str]:
    """Keep case, punctuation, and line breaks as potential style signals."""
    return TOKEN_PATTERN.findall(text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"LLM_name", "LLM_Input", "LLM_output", "source_id", "domain"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No examples in {path}")
    if set(row["LLM_name"] for row in rows) != set(LABELS):
        raise ValueError(f"Expected labels {LABELS} in {path}")
    return rows


def build_vocabulary(rows: list[dict[str, str]], max_size: int, min_frequency: int) -> dict[str, int]:
    # Fit token IDs on training rows only; validation and test may contain unknown tokens.
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row["LLM_Input"]))
        counts.update(tokenize(row["LLM_output"]))
    # Reserve IDs for padding, unseen tokens, and the prompt/response boundary.
    ordered = sorted(
        (token for token, count in counts.items() if count >= min_frequency),
        key=lambda token: (-counts[token], token),
    )[:max_size - 3]
    return {token: index for index, token in enumerate((PAD, UNK, SEP, *ordered))}


def input_tokens(row: dict[str, str], mode: str, prompt_limit: int, output_limit: int) -> list[str]:
    # Apply the same limits in all splits and separate the two fields when combined.
    prompt = tokenize(row["LLM_Input"])[:prompt_limit]
    output = tokenize(row["LLM_output"])[:output_limit]
    if mode == "input":
        return prompt
    if mode == "output":
        return output
    return prompt + [SEP] + output


class TextDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, str]], vocabulary: dict[str, int], mode: str,
        prompt_limit: int, output_limit: int,
    ) -> None:
        self.samples = []
        label_ids = {label: index for index, label in enumerate(LABELS)}
        for row in rows:
            tokens = input_tokens(row, mode, prompt_limit, output_limit)
            if not tokens:
                raise ValueError(f"Empty {mode} text in record {row.get('record_id', '')}")
            # Validation and test tokens absent from training map to <unk>.
            ids = [vocabulary.get(token, vocabulary[UNK]) for token in tokens]
            self.samples.append((torch.tensor(ids, dtype=torch.long), label_ids[row["LLM_name"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)
    # Save original lengths before padding so both models can ignore pad tokens.
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    # The widest CNN filter needs at least five token positions.
    if padded.shape[1] < 5:
        padded = nn.functional.pad(padded, (0, 5 - padded.shape[1]))
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        # Different filter widths detect short patterns of three to five tokens.
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=width) for width in (3, 4, 5)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, len(LABELS))

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Conv1d expects embedding features in the channel dimension.
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            # Ignore padded positions before max pooling over each filter's matches.
            valid_widths = (lengths - convolution.kernel_size[0] + 1).clamp(min=1)
            positions = torch.arange(features.shape[-1], device=features.device)
            features = features.masked_fill(positions[None, None, :] >= valid_widths[:, None, None], float("-inf"))
            pooled.append(features.max(dim=-1).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        # Read the sequence in both directions before combining final states.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, len(LABELS))

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        # Packing skips padding; the two final states summarize both directions.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))


def metric_summary(confusion: list[list[int]]) -> dict:
    # Confusion rows are true labels; columns are predicted labels.
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(LABELS)))
    precision, recall, f1 = [], [], []
    for index in range(len(LABELS)):
        true_positive = confusion[index][index]
        predicted = sum(confusion[row][index] for row in range(len(LABELS)))
        actual = sum(confusion[index])
        p = true_positive / predicted if predicted else 0.0
        r = true_positive / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    # Macro scores weight each family equally, even if class sizes differ.
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precision) / len(LABELS),
        "macro_recall": sum(recall) / len(LABELS),
        "macro_f1": sum(f1) / len(LABELS),
        "per_class_precision": dict(zip(LABELS, precision)),
        "per_class_recall": dict(zip(LABELS, recall)),
        "per_class_f1": dict(zip(LABELS, f1)),
        "confusion_matrix": confusion,
        "confusion_labels": LABELS,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    total_loss = 0.0
    # Sum batch losses first, then divide once by the number of examples.
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for token_ids, lengths, labels in loader:
        token_ids, lengths, labels = token_ids.to(device), lengths.to(device), labels.to(device)
        logits = model(token_ids, lengths)
        total_loss += criterion(logits, labels).item()
        for actual, predicted in zip(labels.tolist(), logits.argmax(dim=1).tolist()):
            confusion[actual][predicted] += 1
    metrics = metric_summary(confusion)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_one(
    architecture: str, mode: str, rows: dict, vocabulary: dict, args: argparse.Namespace,
    device: torch.device,
) -> dict:
    run_name = f"{architecture}_{mode}"
    run_dir = args.output_dir / run_name
    # Keep each architecture and input setting in its own run directory.
    if run_dir.exists():
        raise FileExistsError(f"Refusing to replace previous run: {run_dir}")
    run_dir.mkdir(parents=True)

    datasets = {
        split: TextDataset(split_rows, vocabulary, mode, args.prompt_limit, args.output_limit)
        for split, split_rows in rows.items()
    }
    # Shuffle training batches only; evaluation order has no effect on metrics.
    train_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == "train"),
            generator=train_generator if split == "train" else None,
            collate_fn=collate_batch, num_workers=0, pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    if architecture == "cnn":
        model = TextCNN(len(vocabulary), args.embedding_dim, args.cnn_channels, args.dropout)
    else:
        model = BiLSTM(len(vocabulary), args.embedding_dim, args.lstm_hidden, args.dropout)
    model.to(device)
    # Cross-entropy measures label errors; AdamW applies the weight updates.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for token_ids, lengths, labels in loaders["train"]:
            token_ids, lengths, labels = token_ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(token_ids, lengths), labels)
            loss.backward()
            # Clipping limits unstable gradient spikes, especially in the LSTM.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)

        validation = evaluate(model, loaders["validation"], device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(datasets["train"]),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(record)
        print(f"{run_name} epoch {epoch}: train loss {record['train_loss']:.4f}, "
              f"validation macro-F1 {validation['macro_f1']:.4f}", flush=True)
        # Select the checkpoint by validation F1 and stop after stalled epochs.
        if validation["macro_f1"] > best_score + 1e-6:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        elif epoch - best_epoch >= args.patience:
            break

    # The test set is evaluated once, using the validation-selected checkpoint.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, loaders["test"], device)
    result = {
        "architecture": architecture,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "test": test_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{run_name} test accuracy {test_metrics['accuracy']:.4f}, "
          f"macro-F1 {test_metrics['macro_f1']:.4f}", flush=True)
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_saved_runs(args: argparse.Namespace, device: torch.device) -> None:
    config = json.loads((args.output_dir / "configuration.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((args.output_dir / "vocabulary.json").read_text(encoding="utf-8"))
    test_rows = read_rows(args.data_dir / "test.csv")
    architectures = ("cnn", "lstm") if args.architecture == "all" else (args.architecture,)
    modes = ("output", "input", "both") if args.mode == "all" else (args.mode,)

    for mode in modes:
        # Reuse the saved vocabulary and text limits so evaluation matches training.
        dataset = TextDataset(test_rows, vocabulary, mode, config["prompt_limit"], config["output_limit"])
        loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_batch)
        for architecture in architectures:
            if architecture == "cnn":
                model = TextCNN(len(vocabulary), config["embedding_dim"], config["cnn_channels"], config["dropout"])
            else:
                model = BiLSTM(len(vocabulary), config["embedding_dim"], config["lstm_hidden"], config["dropout"])
            # Load the validation-selected weights and score the held-out test rows.
            checkpoint = args.output_dir / f"{architecture}_{mode}" / "best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            metrics = evaluate(model.to(device), loader, device)
            print(json.dumps({"run": f"{architecture}_{mode}", "test": metrics}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/raid_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--architecture", choices=("cnn", "lstm", "all"), default="all")
    parser.add_argument("--mode", choices=("input", "output", "both", "all"), default="all")
    parser.add_argument("--seed", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=96)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--prompt-limit", type=int, default=96)
    parser.add_argument("--output-limit", type=int, default=512)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--evaluate-only", action="store_true", help="Load saved checkpoints and score the test split without training")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.prompt_limit, args.output_limit) <= 0:
        parser.error("Epoch, patience, batch, and text limits must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__}, device {device}", flush=True)
    if args.evaluate_only:
        evaluate_saved_runs(args, device)
        return
    rows = {split: read_rows(args.data_dir / f"{split}.csv") for split in ("train", "validation", "test")}
    vocabulary = build_vocabulary(rows["train"], args.max_vocab, args.min_frequency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output_dir / "vocabulary.json"
    if not vocabulary_path.exists():
        vocabulary_path.write_text(json.dumps(vocabulary, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        if json.loads(vocabulary_path.read_text(encoding="utf-8")) != vocabulary:
            raise ValueError("Stored vocabulary does not match the current training data or settings")

    # Save the settings and source manifest alongside the model outputs.
    configuration = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "torch_version": torch.__version__,
        "device": str(device),
        "vocabulary_size": len(vocabulary),
        "labels": LABELS,
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8")),
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=str) + "\n", encoding="utf-8"
    )
    architectures = ("cnn", "lstm") if args.architecture == "all" else (args.architecture,)
    modes = ("output", "input", "both") if args.mode == "all" else (args.mode,)
    # Reset the seed so every setting starts with the same random state.
    for mode in modes:
        for architecture in architectures:
            set_seed(args.seed)
            train_one(architecture, mode, rows, vocabulary, args, device)


if __name__ == "__main__":
    main()
