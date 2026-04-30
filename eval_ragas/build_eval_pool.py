import argparse
from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = [
    "sample_id",
    "question",
    "reference",
    "source_type",
    "theme",
    "topic",
    "difficulty",
    "tags",
    "notes",
]


def load_one_csv(path: Path, default_source_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "question" not in df.columns:
        raise ValueError(f"{path} must contain at least a 'question' column.")

    if "reference" not in df.columns:
        df["reference"] = ""

    if "sample_id" not in df.columns:
        prefix = "g" if default_source_type == "golden" else "s"
        df["sample_id"] = [f"{prefix}_{i:04d}" for i in range(1, len(df) + 1)]

    if "source_type" not in df.columns:
        df["source_type"] = default_source_type

    for col in ["theme", "topic", "difficulty", "tags", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df = df[STANDARD_COLUMNS].copy()
    df["question"] = df["question"].fillna("").astype(str).str.strip()
    df["reference"] = df["reference"].fillna("").astype(str).str.strip()
    df["source_type"] = df["source_type"].fillna(default_source_type).astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge golden and synthetic QA into one evaluation pool.")
    parser.add_argument("--golden", type=str, default="data_eval/golden_qa.csv")
    parser.add_argument("--synthetic", type=str, default="data_eval/synthetic_qa.csv")
    parser.add_argument("--out", type=str, default="data_eval/eval_pool.csv")
    args = parser.parse_args()

    frames = []

    golden_path = Path(args.golden)
    if golden_path.exists():
        frames.append(load_one_csv(golden_path, "golden"))

    synthetic_path = Path(args.synthetic)
    if synthetic_path.exists():
        frames.append(load_one_csv(synthetic_path, "synthetic"))

    if not frames:
        raise FileNotFoundError("No input CSV found. Provide at least one of --golden or --synthetic.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["question", "reference"], keep="first")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"Saved evaluation pool to: {out_path.resolve()}")
    print(f"Rows: {len(merged)}")
    print(merged["source_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()