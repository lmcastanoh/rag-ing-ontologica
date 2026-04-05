# backend/evaluation.py
# ==============================================================================
# Modulo de evaluacion del sistema RAG.
#
# Metricas de retrieval (requieren ground truth):
#   - recall_at_k:    fraccion de docs relevantes recuperados
#   - precision_at_k: fraccion de docs recuperados que son relevantes
#   - reciprocal_rank: posicion del primer doc relevante (para MRR)
#   - ndcg_at_k:      relevancia ponderada por posicion
#
# Metricas LLM-as-Judge (sin ground truth):
#   - judge_relevance:   la respuesta es relevante para la pregunta?
#   - judge_faithfulness: la respuesta es fiel al contexto recuperado?
# ==============================================================================
from __future__ import annotations

import json
import math
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts import (
    FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
    FAITHFULNESS_JUDGE_USER_TEMPLATE,
    RELEVANCE_JUDGE_SYSTEM_PROMPT,
    RELEVANCE_JUDGE_USER_TEMPLATE,
)


# ── Metricas de Retrieval ───────────────────────────────────────────────────


def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int | None = None,
) -> float:
    """Recall@k: fraccion de documentos relevantes que fueron recuperados.

    Args:
        retrieved_doc_ids: IDs de docs recuperados (en orden de retrieval).
        relevant_doc_ids:  IDs de docs relevantes (ground truth).
        k: Numero de docs a considerar. None = todos los recuperados.

    Returns:
        Recall entre 0.0 y 1.0.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k] if k else retrieved_doc_ids
    retrieved_set = set(top_k)
    relevant_set = set(relevant_doc_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int | None = None,
) -> float:
    """Precision@k: fraccion de documentos recuperados que son relevantes.

    Args:
        retrieved_doc_ids: IDs de docs recuperados (en orden de retrieval).
        relevant_doc_ids:  IDs de docs relevantes (ground truth).
        k: Numero de docs a considerar. None = todos los recuperados.

    Returns:
        Precision entre 0.0 y 1.0.
    """
    top_k = retrieved_doc_ids[:k] if k else retrieved_doc_ids
    if not top_k:
        return 0.0
    retrieved_set = set(top_k)
    relevant_set = set(relevant_doc_ids)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def reciprocal_rank(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Reciprocal Rank: 1/posicion del primer documento relevante.

    Usado para calcular MRR (Mean Reciprocal Rank) promediando sobre queries.

    Args:
        retrieved_doc_ids: IDs de docs recuperados (en orden).
        relevant_doc_ids:  IDs de docs relevantes (ground truth).

    Returns:
        1/(posicion+1) del primer doc relevante, o 0.0 si ninguno es relevante.
    """
    relevant_set = set(relevant_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int | None = None,
) -> float:
    """nDCG@k: Normalized Discounted Cumulative Gain.

    Mide la calidad del ranking ponderando relevancia por posicion.
    Relevancia binaria: 1 si el doc es relevante, 0 si no.

    DCG@k  = sum(rel_i / log2(i + 2))  para i = 0..k-1
    IDCG@k = DCG ideal (todos los relevantes al inicio)
    nDCG   = DCG / IDCG

    Args:
        retrieved_doc_ids: IDs de docs recuperados (en orden).
        relevant_doc_ids:  IDs de docs relevantes (ground truth).
        k: Numero de posiciones a evaluar. None = todos.

    Returns:
        nDCG entre 0.0 y 1.0.
    """
    relevant_set = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k] if k else retrieved_doc_ids

    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 porque log2(1) = 0

    # IDCG: DCG ideal (todos los relevantes primero)
    n_relevant = min(len(relevant_set), len(top_k))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int | None = None,
) -> dict[str, float]:
    """Calcula todas las metricas de retrieval en un solo paso.

    Returns:
        Dict con recall_at_k, precision_at_k, mrr, ndcg_at_k, k.
    """
    effective_k = k or len(retrieved_doc_ids)
    return {
        "recall_at_k": round(recall_at_k(retrieved_doc_ids, relevant_doc_ids, k), 4),
        "precision_at_k": round(precision_at_k(retrieved_doc_ids, relevant_doc_ids, k), 4),
        "mrr": round(reciprocal_rank(retrieved_doc_ids, relevant_doc_ids), 4),
        "ndcg_at_k": round(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k), 4),
        "k": effective_k,
    }


# ── LLM-as-Judge ────────────────────────────────────────────────────────────


def _get_judge_llm() -> ChatOpenAI:
    """LLM para evaluacion (gpt-5-nano, temperature=0 para consistencia)."""
    return ChatOpenAI(model="gpt-5-nano", temperature=0)


def _parse_json_response(text: str) -> dict:
    """Intenta extraer JSON de la respuesta del LLM."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Buscar JSON embebido en texto
        import re
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def judge_relevance(question: str, answer: str) -> dict[str, Any]:
    """LLM-as-Judge: evalua si la respuesta es relevante para la pregunta.

    Args:
        question: Pregunta original del usuario.
        answer:   Respuesta generada por el RAG.

    Returns:
        Dict con score (0.0-1.0) y justification.
    """
    llm = _get_judge_llm()
    response = llm.invoke(
        [
            SystemMessage(content=RELEVANCE_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=RELEVANCE_JUDGE_USER_TEMPLATE.format(
                question=question,
                answer=answer,
            )),
        ]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    result = _parse_json_response(text)
    return {
        "relevance_score": round(float(result.get("score", 0.0)), 4),
        "relevance_justification": result.get("justification", text[:200]),
    }


def judge_faithfulness(question: str, context: str, answer: str) -> dict[str, Any]:
    """LLM-as-Judge: evalua si la respuesta es fiel al contexto (no alucina).

    Args:
        question: Pregunta original del usuario.
        context:  Contexto recuperado (chunks concatenados).
        answer:   Respuesta generada por el RAG.

    Returns:
        Dict con score, supported_claims, total_claims, unsupported_claims.
    """
    llm = _get_judge_llm()
    response = llm.invoke(
        [
            SystemMessage(content=FAITHFULNESS_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=FAITHFULNESS_JUDGE_USER_TEMPLATE.format(
                question=question,
                context=context,
                answer=answer,
            )),
        ]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    result = _parse_json_response(text)
    return {
        "faithfulness_score": round(float(result.get("score", 0.0)), 4),
        "supported_claims": int(result.get("supported_claims", 0)),
        "total_claims": int(result.get("total_claims", 0)),
        "unsupported_claims": result.get("unsupported", []),
    }


def compute_llm_judge_metrics(
    question: str, context: str, answer: str,
) -> dict[str, Any]:
    """Ejecuta ambas evaluaciones LLM-as-Judge y retorna metricas combinadas."""
    relevance = judge_relevance(question, answer)
    faithfulness = judge_faithfulness(question, context, answer)
    return {**relevance, **faithfulness}
