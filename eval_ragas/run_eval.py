import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import EvaluationDataset, aevaluate
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    BleuScore,
    RougeScore,
)

# Depending on your installed Ragas version, AnswerAccuracy may move.
# If this import fails, replace it with the closest metric available in your version.
from ragas.metrics.collections import AnswerAccuracy

from local_rag_adapter import LocalRAGClient


def build_metrics() -> Tuple[List[Any], List[Any]]:
    """
    Returns:
    - metrics_with_reference: for rows that have a gold reference answer
    - metrics_without_reference: for rows without reference
    """

    grounded_or_honest_refusal = DiscreteMetric(
        name="grounded_or_honest_refusal",
        prompt=(
            "Evaluate the answer.\n"
            "A good answer must either:\n"
            "1) answer correctly from the retrieved contexts, or\n"
            "2) clearly say that the information is missing or insufficient instead of inventing facts.\n\n"
            "Question: {user_input}\n"
            "Retrieved Contexts: {retrieved_contexts}\n"
            "Response: {response}\n\n"
            "Return only 'pass' or 'fail'."
        ),
        allowed_values=["pass", "fail"],
    )

    metrics_with_reference = [
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
        AnswerAccuracy(),
        BleuScore(),
        RougeScore(rouge_type="rougeL", mode="fmeasure"),
        grounded_or_honest_refusal,
    ]

    metrics_without_reference = [
        Faithfulness(),
        grounded_or_honest_refusal,
    ]

    return metrics_with_reference, metrics_without_reference


async def collect_rag_outputs(
    pool_df: pd.DataFrame,
    rag_top_k: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Calls the local RAG once per question and returns:
    - ragas_rows: rows in Ragas-compatible format
    - audit_df: richer metadata for debugging and traceability
    """

    ragas_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    async with LocalRAGClient() as rag_client:
        for _, row in pool_df.iterrows():
            question = str(row["question"]).strip()
            reference = (
                str(row["reference"]).strip()
                if pd.notna(row["reference"])
                else ""
            )

            rag_response = await rag_client.aquery(
                question=question,
                top_k=rag_top_k,
            )

            ragas_rows.append(
                {
                    "user_input": question,
                    "response": rag_response.answer,
                    "retrieved_contexts": rag_response.retrieved_contexts,
                    "retrieved_context_ids": rag_response.retrieved_context_ids,
                    "reference": reference if reference else None,
                }
            )

            audit_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "source_type": row.get("source_type", ""),
                    "topic": row.get("topic", ""),
                    "difficulty": row.get("difficulty", ""),
                    "tags": row.get("tags", ""),
                    "notes": row.get("notes", ""),
                    "question": question,
                    "reference": reference,
                    "response": rag_response.answer,
                    "retrieved_contexts": json.dumps(
                        rag_response.retrieved_contexts,
                        ensure_ascii=False,
                    ),
                    "retrieved_context_ids": json.dumps(
                        rag_response.retrieved_context_ids,
                        ensure_ascii=False,
                    ),
                    "latency_ms": rag_response.latency_ms,
                    "raw_payload": json.dumps(
                        rag_response.raw_payload,
                        ensure_ascii=False,
                    ),
                }
            )

    return ragas_rows, pd.DataFrame(audit_rows)


def split_rows_by_reference(
    ragas_rows: List[Dict[str, Any]],
    audit_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    with_ref_idx = []
    without_ref_idx = []

    for i, row in enumerate(ragas_rows):
        ref = row.get("reference")
        if ref is None or str(ref).strip() == "":
            without_ref_idx.append(i)
        else:
            with_ref_idx.append(i)

    rows_with_ref = [ragas_rows[i] for i in with_ref_idx]
    rows_without_ref = [ragas_rows[i] for i in without_ref_idx]

    audit_with_ref = audit_df.iloc[with_ref_idx].reset_index(drop=True)
    audit_without_ref = audit_df.iloc[without_ref_idx].reset_index(drop=True)

    return rows_with_ref, audit_with_ref, rows_without_ref, audit_without_ref


async def evaluate_subset(
    rows: List[Dict[str, Any]],
    audit_df: pd.DataFrame,
    metrics: List[Any],
    judge_model: str,
    openai_api_key: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    dataset = EvaluationDataset.from_list(rows)

    judge_client = AsyncOpenAI(api_key=openai_api_key)
    judge_llm = llm_factory(judge_model, client=judge_client)

    result = await aevaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
    )

    result_df = result.to_pandas().reset_index(drop=True)

    merged = pd.concat([audit_df.reset_index(drop=True), result_df], axis=1)
    return merged


async def main() -> None:
    load_dotenv()

    openai_api_key = os.environ["OPENAI_API_KEY"]
    judge_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    rag_top_k = int(os.getenv("LOCAL_RAG_TOP_K", "5"))

    pool_path = Path("data_eval/eval_pool.csv")
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Evaluation pool not found: {pool_path.resolve()}"
        )

    pool_df = pd.read_csv(pool_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_with_reference, metrics_without_reference = build_metrics()

    ragas_rows, audit_df = await collect_rag_outputs(
        pool_df=pool_df,
        rag_top_k=rag_top_k,
    )

    audit_df.to_csv(out_dir / "raw_rag_outputs.csv", index=False)

    rows_with_ref, audit_with_ref, rows_without_ref, audit_without_ref = split_rows_by_reference(
        ragas_rows=ragas_rows,
        audit_df=audit_df,
    )

    scored_with_ref = await evaluate_subset(
        rows=rows_with_ref,
        audit_df=audit_with_ref,
        metrics=metrics_with_reference,
        judge_model=judge_model,
        openai_api_key=openai_api_key,
    )
    if not scored_with_ref.empty:
        scored_with_ref.to_csv(out_dir / "scored_with_reference.csv", index=False)

    scored_without_ref = await evaluate_subset(
        rows=rows_without_ref,
        audit_df=audit_without_ref,
        metrics=metrics_without_reference,
        judge_model=judge_model,
        openai_api_key=openai_api_key,
    )
    if not scored_without_ref.empty:
        scored_without_ref.to_csv(out_dir / "scored_without_reference.csv", index=False)

    summary_rows = []

    if not scored_with_ref.empty:
        numeric_cols = [
            col
            for col in [
                "faithfulness",
                "context_precision",
                "context_recall",
                "answer_accuracy",
                "bleu_score",
                "rouge_score",
                "latency_ms",
            ]
            if col in scored_with_ref.columns
        ]

        for col in numeric_cols:
            summary_rows.append(
                {
                    "subset": "with_reference",
                    "metric": col,
                    "mean": scored_with_ref[col].mean(),
                    "median": scored_with_ref[col].median(),
                    "min": scored_with_ref[col].min(),
                    "max": scored_with_ref[col].max(),
                }
            )

    if not scored_without_ref.empty:
        numeric_cols = [
            col
            for col in ["faithfulness", "latency_ms"]
            if col in scored_without_ref.columns
        ]

        for col in numeric_cols:
            summary_rows.append(
                {
                    "subset": "without_reference",
                    "metric": col,
                    "mean": scored_without_ref[col].mean(),
                    "median": scored_without_ref[col].median(),
                    "min": scored_without_ref[col].min(),
                    "max": scored_without_ref[col].max(),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print(f"Done. Outputs written to: {out_dir.resolve()}")
    if not summary_df.empty:
        print(summary_df)


if __name__ == "__main__":
    asyncio.run(main())