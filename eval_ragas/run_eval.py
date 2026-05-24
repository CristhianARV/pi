import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import inspect
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    BleuScore,
    RougeScore,
)

def normalize_theme(theme: str | None) -> str | None:
    if theme is None:
        return None

    value = str(theme).strip()
    if not value:
        return None

    if value.lower() in {"unknown", "none", "nan", "null"}:
        return None

    canonical_themes = {
        "commissioning": "Commissioning",
        "common_documents": "Common_documents",
        "common documents": "Common_documents",
        "cycles": "Cycles",
        "functional_description": "Functional_description",
        "functional description": "Functional_description",
        "interfaces": "Interfaces",
        "manuals": "Manuals",
        "plc-libraries": "PLC-libraries",
        "plc_libraries": "PLC-libraries",
        "plc libraries": "PLC-libraries",
        "troubleshooting": "troubleshooting",
        "operator_support": "operator_support",
        "operator support": "operator_support",
        "safety_guardrail": "safety_guardrail",
        "safety guardrail": "safety_guardrail",
        "unanswerable": "unanswerable",
    }

    return canonical_themes.get(value.lower(), value)

# Depending on your installed Ragas version, AnswerAccuracy may move.
# If this import fails, replace it with the closest metric available in your version.
from ragas.metrics.collections import AnswerAccuracy

from local_rag_adapter import LocalRAGClient


def build_metrics(judge_llm) -> Tuple[List[Any], List[Any]]:
    """
    Returns:
    - metrics_with_reference: for rows that have a gold reference answer
    - metrics_without_reference: for rows without reference
    """

    coverage_metric = DiscreteMetric(
        name="coverage",
        prompt=(
            "Evaluate how completely the response covers the expected answer.\n"
            "Question: {user_input}\n"
            "Reference: {reference}\n"
            "Response: {response}\n\n"
            "Return only one of: 'complete', 'partial', 'missing'."
        ),
        allowed_values=["complete", "partial", "missing"],
    )

    clarity_metric = DiscreteMetric(
        name="clarity",
        prompt=(
            "Evaluate the clarity of the response, considering that it must be understandable "
            "by a human operator in a production environment.\n"
            "A response is:\n"
            "- 'clear' if it is easy to understand, precise, and directly usable;\n"
            "- 'acceptable' if it is globally understandable but could be clearer or better structured;\n"
            "- 'unclear' if it is confusing, ambiguous, or difficult to use.\n"
            "Response: {response}\n"
            "Return only one of: 'clear', 'acceptable', 'unclear'."
        ),
        allowed_values=["clear", "acceptable", "unclear"],
    )

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

    coverage_metric.llm = judge_llm
    clarity_metric.llm = judge_llm
    grounded_or_honest_refusal.llm = judge_llm

    metrics_with_reference = [
        coverage_metric,
        clarity_metric,
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerAccuracy(llm=judge_llm, name="answer_accuracy"),
        BleuScore(),
        RougeScore(rouge_type="rougeL", mode="fmeasure"),
        grounded_or_honest_refusal,
    ]

    metrics_without_reference = [
        clarity_metric,
        Faithfulness(llm=judge_llm),
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

            theme = None
            if "theme" in row and pd.notna(row["theme"]):
                theme = str(row["theme"]).strip() or None

            rag_response = await rag_client.aquery(
                question=question,
                top_k=rag_top_k,
                theme=theme,
            )

            retrieved_context_metadata = rag_response.retrieved_context_metadata

            ragas_rows.append(
                {
                    "user_input": question,
                    "response": rag_response.answer,
                    "retrieved_contexts": rag_response.retrieved_contexts,
                    "retrieved_context_ids": rag_response.retrieved_context_ids,
                    "reference": reference if reference else None,
                    "selected_theme": theme,
                }
            )

            audit_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "source_type": row.get("source_type", ""),
                    "theme": row.get("theme", ""),
                    "selected_theme": theme,
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

def _extract_metric_value(result: Any) -> Any:
    """
    Normalize various Ragas metric result shapes to a scalar value.
    Works for numeric metrics and custom DiscreteMetric outputs.
    """
    if result is None:
        return None

    if isinstance(result, (str, int, float, bool)):
        return result

    if isinstance(result, dict):
        for key in ("score", "value", "label", "verdict"):
            if key in result and result[key] is not None:
                return result[key]

    for attr in ("score", "value", "label", "verdict"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                return val

    if hasattr(result, "model_dump"):
        data = result.model_dump()
        if isinstance(data, dict):
            for key in ("score", "value", "label", "verdict"):
                if key in data and data[key] is not None:
                    return data[key]

    return str(result)

async def evaluate_subset(
    rows: List[Dict[str, Any]],
    audit_df: pd.DataFrame,
    metrics: List[Any],
    judge_llm,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    scored_rows = []

    for row in rows:
        metric_results: Dict[str, Any] = {}

        available_values = {
            "user_input": row.get("user_input"),
            "response": row.get("response"),
            "reference": row.get("reference"),
            "retrieved_contexts": row.get("retrieved_contexts"),
            "retrieved_context_ids": row.get("retrieved_context_ids"),
        }

        for metric in metrics:
            metric_name = getattr(metric, "name", type(metric).__name__)

            try:
                if hasattr(metric, "llm") and getattr(metric, "llm", None) is None:
                    metric.llm = judge_llm

                sig = inspect.signature(metric.ascore)
                accepted_params = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    if param.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        accepted_params.append(param_name)

                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        accepted_params = list(available_values.keys())
                        break

                kwargs = {
                    k: v
                    for k, v in available_values.items()
                    if k in accepted_params and v is not None
                }

                if metric_name in {
                    "coverage",
                    "clarity",
                    "grounded_or_honest_refusal",
                }:
                    kwargs["llm"] = judge_llm

                result = await metric.ascore(**kwargs)
                metric_results[metric_name] = _extract_metric_value(result)

            except Exception as e:
                metric_results[metric_name] = None
                metric_results[f"{metric_name}_error"] = str(e)

        scored_rows.append(metric_results)

    result_df = pd.DataFrame(scored_rows).reset_index(drop=True)
    merged = pd.concat([audit_df.reset_index(drop=True), result_df], axis=1)

    return merged


async def main() -> None:
    load_dotenv()

    openai_api_key = os.environ["OPENAI_API_KEY"]
    judge_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    rag_top_k = int(os.getenv("LOCAL_RAG_TOP_K", "5"))
    judge_client = AsyncOpenAI(api_key=openai_api_key)
    judge_llm = llm_factory(judge_model, client=judge_client)

    pool_path = Path(os.getenv("EVAL_POOL_PATH", "data_eval/eval_pool.csv"))
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Evaluation pool not found: {pool_path.resolve()}"
        )

    pool_df = pd.read_csv(pool_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_with_reference, metrics_without_reference = build_metrics(judge_llm)

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
        judge_llm=judge_llm,
    )
    if not scored_with_ref.empty:
        scored_with_ref.to_csv(out_dir / "scored_with_reference.csv", index=False)

    scored_without_ref = await evaluate_subset(
        rows=rows_without_ref,
        audit_df=audit_without_ref,
        metrics=metrics_without_reference,
        judge_llm=judge_llm,
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

        for col in ["coverage", "clarity", "grounded_or_honest_refusal"]:
            if col in scored_with_ref.columns:
                counts = scored_with_ref[col].value_counts(dropna=False)
                for value, count in counts.items():
                    summary_rows.append(
                        {
                            "subset": "with_reference",
                            "metric": f"{col}={value}",
                            "mean": count,
                            "median": None,
                            "min": None,
                            "max": None,
                            "count": count,
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