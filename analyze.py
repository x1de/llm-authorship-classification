"""Measure response fingerprints and simple output-only baselines."""

import argparse
import csv
import json
import math
import re
import statistics
import string
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import LABELS, tokenize


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style_features(text: str) -> list[float]:
    # These eight values describe response form without using the model label.
    words = WORD_PATTERN.findall(text)
    token_count = len(tokenize(text))
    word_count = len(words)
    punctuation = sum(character in string.punctuation for character in text)
    uppercase_words = sum(word.isupper() and len(word) > 1 for word in words)
    # Log counts reduce the influence of unusually long responses.
    return [
        math.log1p(token_count),
        math.log1p(word_count),
        sum(map(len, words)) / max(word_count, 1),
        len({word.lower() for word in words}) / max(word_count, 1),
        punctuation / max(word_count, 1),
        math.log1p(text.count("\n")),
        uppercase_words / max(word_count, 1),
        sum(character.isdigit() for character in text) / max(len(text), 1),
    ]


def summarize(true_labels, predicted_labels) -> dict:
    # Macro averages give all four families the same weight.
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_labels": LABELS,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=LABELS).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/raid_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/raid_v1/baselines_v3.json"))
    args = parser.parse_args()
    rows = {split: load_rows(args.data_dir / f"{split}.csv") for split in ("train", "test")}
    labels = {split: [row["LLM_name"] for row in examples] for split, examples in rows.items()}
    texts = {split: [row["LLM_output"] for row in examples] for split, examples in rows.items()}
    features = {split: [style_features(text) for text in samples] for split, samples in texts.items()}

    # Fit each exploratory classifier on training responses only.
    length_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=448))
    length_model.fit([[item[0]] for item in features["train"]], labels["train"])
    length_result = summarize(labels["test"], length_model.predict([[item[0]] for item in features["test"]]))

    style_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=448))
    style_model.fit(features["train"], labels["train"])
    style_result = summarize(labels["test"], style_model.predict(features["test"]))

    # Remove response length and line breaks to check their contribution.
    no_length_columns = (2, 3, 4, 6, 7)
    no_length_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=448))
    no_length_model.fit(
        [[item[index] for index in no_length_columns] for item in features["train"]], labels["train"]
    )
    no_length_result = summarize(
        labels["test"],
        no_length_model.predict([[item[index] for index in no_length_columns] for item in features["test"]]),
    )

    # Word and two-word phrase features provide a content-based baseline.
    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, random_state=448),
    )
    tfidf_model.fit(texts["train"], labels["train"])
    tfidf_result = summarize(labels["test"], tfidf_model.predict(texts["test"]))
    # Inspect training-fitted term weights as associations, not causal features.
    vectorizer = tfidf_model.named_steps["tfidfvectorizer"]
    classifier = tfidf_model.named_steps["logisticregression"]
    terms = vectorizer.get_feature_names_out()
    top_terms = {
        label: [str(terms[index]) for index in classifier.coef_[class_index].argsort()[-10:][::-1]]
        for class_index, label in enumerate(classifier.classes_)
    }

    # Describe training responses only; the test split is reserved for scores.
    by_label = defaultdict(list)
    for row in rows["train"]:
        by_label[row["LLM_name"]].append(row["LLM_output"])
    descriptive = {}
    for label in LABELS:
        passages = by_label[label]
        tokens = [len(tokenize(text)) for text in passages]
        newlines = [text.count("\n") for text in passages]
        descriptive[label] = {
            "training_rows": len(passages),
            "median_tokens": statistics.median(tokens),
            "median_newlines": statistics.median(newlines),
            "median_lexical_diversity": statistics.median(style_features(text)[3] for text in passages),
            "median_punctuation_per_word": statistics.median(style_features(text)[4] for text in passages),
        }

    # Compare response lengths within the same source prompt to control for topic.
    paired = defaultdict(dict)
    for row in rows["train"]:
        paired[row["source_id"]][row["LLM_name"]] = len(tokenize(row["LLM_output"]))
    paired_length = {}
    for first, second in (("ChatGPT", "Cohere"), ("Llama", "Mistral")):
        counts = [(group[first], group[second]) for group in paired.values() if first in group and second in group]
        paired_length[f"{first}_longer_than_{second}"] = {
            "matched_prompts": len(counts),
            "fraction": sum(left > right for left, right in counts) / len(counts),
        }

    results = {
        "training_only_fit": True,
        "length_only_logistic_regression": length_result,
        "eight_style_features_logistic_regression": style_result,
        "style_without_length_or_line_breaks_logistic_regression": no_length_result,
        "tfidf_unigram_bigram_logistic_regression": tfidf_result,
        "top_positive_tfidf_terms_by_class": top_terms,
        "style_feature_names": [
            "log_token_count", "log_word_count", "mean_word_length", "lexical_diversity",
            "punctuation_per_word", "log_newline_count", "uppercase_word_fraction", "digit_fraction",
        ],
        "training_descriptives": descriptive,
        "paired_length_comparison": paired_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep prior analysis outputs rather than overwriting measured results.
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing results: {args.output}")
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name in ("length_only_logistic_regression", "eight_style_features_logistic_regression",
                 "style_without_length_or_line_breaks_logistic_regression",
                 "tfidf_unigram_bigram_logistic_regression"):
        print(name, "accuracy", round(results[name]["accuracy"], 4),
              "macro-F1", round(results[name]["macro_f1"], 4))


if __name__ == "__main__":
    main()
