"""
Generate document-ready evaluation artifacts from test tweet files.

Outputs:
- Per-tweet predictions CSV
- Metrics tables (classification report, confusion matrix, summary stats)
- Charts (class distribution, confidence histogram, confusion matrix, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Ensure project root is importable when script is run from reports/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dashboard.config import ACTIONABLE_LABELS, LABEL_DISPLAY_NAMES, LABEL_MAP
from Dashboard.model_inference import ModelInference


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
ALL_LABELS = sorted(LABEL_MAP.keys())


def parse_test_tweets_file(file_path: Path) -> pd.DataFrame:
    """Parse tweets and optional expected labels from the structured test file."""
    category_re = re.compile(r"^#\s*CATEGORY\s+(\d+)\s*:", re.IGNORECASE)
    reset_re = re.compile(
        r"^#\s*(COMPLEX|LOCATION-SPECIFIC|TIME-SPECIFIC|NEEDS-SPECIFIC|PEOPLE COUNT|DAMAGE TYPE|END OF)",
        re.IGNORECASE,
    )

    rows: List[Dict[str, Any]] = []
    current_expected_label: Optional[int] = None

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                category_match = category_re.match(line)
                if category_match:
                    current_expected_label = int(category_match.group(1))
                    continue

                if reset_re.match(line):
                    current_expected_label = None
                continue

            rows.append(
                {
                    "tweet_idx": len(rows) + 1,
                    "line_no": line_no,
                    "text": line,
                    "expected_label_id": current_expected_label,
                    "expected_label_name": LABEL_DISPLAY_NAMES.get(current_expected_label)
                    if current_expected_label is not None
                    else None,
                }
            )

    return pd.DataFrame(rows)


def format_top_tokens(explanation: Any, k: int = 5) -> str:
    """Return top-k important tokens as a compact string."""
    if not isinstance(explanation, list):
        return ""

    pairs: List[tuple[str, float]] = []
    for item in explanation:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token, score = item[0], item[1]
        elif isinstance(item, dict):
            token, score = item.get("token"), item.get("score")
        else:
            continue

        if token is None:
            continue
        token = str(token)
        if token in SPECIAL_TOKENS:
            continue

        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue
        pairs.append((token, score_val))

    if not pairs:
        return ""

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:k]
    return ", ".join(f"{tok}:{val:.3f}" for tok, val in top)


def list_or_empty(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def actionable_field_lengths(actionable_info: Any) -> Dict[str, int]:
    """Get per-field lengths for actionable info."""
    if not isinstance(actionable_info, dict):
        return {
            "locations_count": 0,
            "needs_count": 0,
            "damage_count": 0,
            "people_count_items": 0,
            "time_mentions_count": 0,
        }

    locations = list_or_empty(actionable_info.get("locations"))
    needs = list_or_empty(actionable_info.get("needs"))
    damage = list_or_empty(actionable_info.get("damageType") or actionable_info.get("damage_type"))
    people_count = list_or_empty(actionable_info.get("peopleCount") or actionable_info.get("people_count"))
    time_mentions = list_or_empty(actionable_info.get("timeMentions") or actionable_info.get("time_mentions"))

    return {
        "locations_count": len(locations),
        "needs_count": len(needs),
        "damage_count": len(damage),
        "people_count_items": len(people_count),
        "time_mentions_count": len(time_mentions),
    }


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    model = ModelInference()
    if not model.loaded:
        raise RuntimeError("Model failed to load. Cannot generate report.")

    out_rows: List[Dict[str, Any]] = []
    total = len(df)

    for idx, row in df.iterrows():
        text = row["text"]
        result = model.classify_tweet(text)
        if result is None:
            pred_label_id = None
            pred_label_name = None
            confidence = None
            top_tokens = ""
            actionable_info = {}
        else:
            pred_label_id = result.get("predicted_label_id")
            pred_label_name = LABEL_DISPLAY_NAMES.get(pred_label_id, result.get("predicted_label"))
            scores = result.get("confidence_scores") or []
            confidence = max(scores) if scores else None
            top_tokens = format_top_tokens(result.get("explanation"), k=5)
            actionable_info = result.get("actionable_info") or {}

        field_lengths = actionable_field_lengths(actionable_info)

        out_rows.append(
            {
                **row.to_dict(),
                "predicted_label_id": pred_label_id,
                "predicted_label_name": pred_label_name,
                "max_confidence": confidence,
                "top_tokens": top_tokens,
                "actionable_info_json": json.dumps(actionable_info, ensure_ascii=False),
                **field_lengths,
            }
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total} tweets")

    return pd.DataFrame(out_rows)


def save_tables_and_figures(results_df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = out_dir / "predictions_per_tweet.csv"
    results_df.to_csv(predictions_csv, index=False, encoding="utf-8")

    labeled_df = results_df.dropna(subset=["expected_label_id"]).copy()
    labeled_df["expected_label_id"] = labeled_df["expected_label_id"].astype(int)
    labeled_df = labeled_df.dropna(subset=["predicted_label_id"]).copy()
    labeled_df["predicted_label_id"] = labeled_df["predicted_label_id"].astype(int)

    summary: Dict[str, Any] = {
        "num_total_tweets": int(len(results_df)),
        "num_labeled_tweets": int(len(labeled_df)),
    }

    # ---------- Summary metrics ----------
    if len(labeled_df) > 0:
        y_true = labeled_df["expected_label_id"]
        y_pred = labeled_df["predicted_label_id"]

        summary["accuracy_labeled"] = float(accuracy_score(y_true, y_pred))
        summary["macro_f1_labeled"] = float(f1_score(y_true, y_pred, average="macro"))
        summary["weighted_f1_labeled"] = float(f1_score(y_true, y_pred, average="weighted"))

        cls_report = classification_report(
            y_true,
            y_pred,
            labels=ALL_LABELS,
            target_names=[LABEL_DISPLAY_NAMES[i] for i in ALL_LABELS],
            output_dict=True,
            zero_division=0,
        )
        cls_df = pd.DataFrame(cls_report).transpose()
        cls_df.to_csv(out_dir / "classification_report_labeled.csv", encoding="utf-8")

        cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
        cm_df = pd.DataFrame(cm, index=[LABEL_DISPLAY_NAMES[i] for i in ALL_LABELS], columns=[LABEL_DISPLAY_NAMES[i] for i in ALL_LABELS])
        cm_df.to_csv(out_dir / "confusion_matrix_labeled.csv", encoding="utf-8")
    else:
        summary["accuracy_labeled"] = None
        summary["macro_f1_labeled"] = None
        summary["weighted_f1_labeled"] = None

    # ---------- Aggregated tables ----------
    predicted_counts = (
        results_df["predicted_label_id"]
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(ALL_LABELS, fill_value=0)
        .rename_axis("label_id")
        .reset_index(name="count")
    )
    predicted_counts["label_name"] = predicted_counts["label_id"].map(LABEL_DISPLAY_NAMES)
    predicted_counts.to_csv(out_dir / "predicted_class_distribution.csv", index=False, encoding="utf-8")

    if len(labeled_df) > 0:
        expected_counts = (
            labeled_df["expected_label_id"]
            .value_counts()
            .reindex(ALL_LABELS, fill_value=0)
            .rename_axis("label_id")
            .reset_index(name="expected_count")
        )
        pred_counts_labeled = (
            labeled_df["predicted_label_id"]
            .value_counts()
            .reindex(ALL_LABELS, fill_value=0)
            .rename_axis("label_id")
            .reset_index(name="predicted_count")
        )
        counts_cmp = expected_counts.merge(pred_counts_labeled, on="label_id")
        counts_cmp["label_name"] = counts_cmp["label_id"].map(LABEL_DISPLAY_NAMES)
        counts_cmp.to_csv(out_dir / "expected_vs_predicted_counts_labeled.csv", index=False, encoding="utf-8")
    else:
        counts_cmp = pd.DataFrame()

    confidence_stats = results_df["max_confidence"].describe().to_frame(name="value")
    confidence_stats.to_csv(out_dir / "confidence_stats.csv", encoding="utf-8")

    actionable_subset = results_df[results_df["predicted_label_id"].isin(ACTIONABLE_LABELS)].copy()
    actionable_coverage = pd.DataFrame(
        {
            "field": ["locations", "needs", "damage", "people_count", "time_mentions"],
            "tweets_with_field": [
                int((actionable_subset["locations_count"] > 0).sum()),
                int((actionable_subset["needs_count"] > 0).sum()),
                int((actionable_subset["damage_count"] > 0).sum()),
                int((actionable_subset["people_count_items"] > 0).sum()),
                int((actionable_subset["time_mentions_count"] > 0).sum()),
            ],
        }
    )
    actionable_coverage.to_csv(out_dir / "actionable_coverage.csv", index=False, encoding="utf-8")

    # ---------- Charts ----------
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=predicted_counts, x="label_name", y="count", palette="Set2")
    plt.xticks(rotation=20, ha="right")
    plt.title("Predicted Class Distribution (All Test Tweets)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_predicted_class_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    conf_vals = results_df["max_confidence"].dropna()
    sns.histplot(conf_vals, bins=15, kde=True, color="#1f77b4")
    plt.title("Confidence Distribution (Max Softmax Probability)")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_confidence_distribution.png", dpi=300)
    plt.close()

    if len(labeled_df) > 0:
        cm = confusion_matrix(labeled_df["expected_label_id"], labeled_df["predicted_label_id"], labels=ALL_LABELS)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[LABEL_DISPLAY_NAMES[i] for i in ALL_LABELS],
            yticklabels=[LABEL_DISPLAY_NAMES[i] for i in ALL_LABELS],
        )
        plt.title("Confusion Matrix (Labeled Subset in test_tweets.txt)")
        plt.xlabel("Predicted")
        plt.ylabel("Expected")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_confusion_matrix_labeled.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        comp = counts_cmp.melt(
            id_vars=["label_id", "label_name"],
            value_vars=["expected_count", "predicted_count"],
            var_name="series",
            value_name="count",
        )
        sns.barplot(data=comp, x="label_name", y="count", hue="series", palette="Set1")
        plt.xticks(rotation=20, ha="right")
        plt.title("Expected vs Predicted Counts (Labeled Subset)")
        plt.xlabel("Class")
        plt.ylabel("Tweet Count")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_expected_vs_predicted_labeled.png", dpi=300)
        plt.close()

    if len(actionable_subset) > 0:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=actionable_coverage, x="field", y="tweets_with_field", palette="viridis")
        plt.title("Actionable Field Coverage (Predicted Actionable Tweets)")
        plt.xlabel("Actionable Field")
        plt.ylabel("Tweets with Non-Empty Field")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_actionable_field_coverage.png", dpi=300)
        plt.close()

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def write_markdown_summary(summary: Dict[str, Any], out_dir: Path, source_file: Path) -> None:
    lines = [
        "# Test Tweets Report",
        "",
        f"- Source file: `{source_file}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Core Metrics",
        "",
        f"- Total test tweets processed: **{summary.get('num_total_tweets', 0)}**",
        f"- Labeled subset size (from CATEGORY 0-4 sections): **{summary.get('num_labeled_tweets', 0)}**",
        f"- Accuracy (labeled subset): **{summary.get('accuracy_labeled')}**",
        f"- Macro F1 (labeled subset): **{summary.get('macro_f1_labeled')}**",
        f"- Weighted F1 (labeled subset): **{summary.get('weighted_f1_labeled')}**",
        "",
        "## Artifacts",
        "",
        "- `predictions_per_tweet.csv`",
        "- `classification_report_labeled.csv`",
        "- `confusion_matrix_labeled.csv`",
        "- `expected_vs_predicted_counts_labeled.csv`",
        "- `predicted_class_distribution.csv`",
        "- `confidence_stats.csv`",
        "- `actionable_coverage.csv`",
        "- `fig_predicted_class_distribution.png`",
        "- `fig_confidence_distribution.png`",
        "- `fig_confusion_matrix_labeled.png`",
        "- `fig_expected_vs_predicted_labeled.png`",
        "- `fig_actionable_field_coverage.png`",
    ]
    (out_dir / "REPORT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation tables/charts from test tweets.")
    parser.add_argument(
        "--input",
        default="test_tweets.txt",
        help="Path to test tweets file (default: test_tweets.txt)",
    )
    parser.add_argument(
        "--out",
        default="reports/test_tweets_report",
        help="Output directory (default: reports/test_tweets_report)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = parse_test_tweets_file(input_path)
    if df.empty:
        raise RuntimeError("No tweet rows found in input file.")

    results_df = run_inference(df)
    summary = save_tables_and_figures(results_df, out_dir)
    write_markdown_summary(summary, out_dir, input_path)

    print("\nReport generation complete.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
