#!/usr/bin/env python
# backend/run_evaluation.py
# ==============================================================================
# Script CLI para evaluacion batch del sistema RAG.
#
# Ejecuta cada pregunta del dataset por el grafo completo y recopila metricas.
# Imprime tabla resumen y guarda resultados en JSON.
#
# Uso: cd backend && python run_evaluation.py
# ==============================================================================
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Cargar .env (un nivel arriba de backend/) — debe ir ANTES de importar
# rag_graph para que LangSmith inicialice el tracing correctamente.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage

from eval_dataset import EVAL_DATASET
from rag_graph import build_rag_graph


async def run_evaluation():
    """Ejecuta evaluacion batch sobre todo el dataset."""
    print("Construyendo grafo RAG...")
    graph = build_rag_graph()
    print(f"Dataset: {len(EVAL_DATASET)} preguntas\n")

    results = []
    retrieval_scores = {"recall": [], "precision": [], "mrr": [], "ndcg": []}
    judge_scores = {"relevance": [], "faithfulness": []}
    grounding_scores = []

    for i, entry in enumerate(EVAL_DATASET):
        question = entry["question"]
        expected_intent = entry["expected_intent"]
        print(f"[{i+1}/{len(EVAL_DATASET)}] {question}")

        inputs = {
            "question": question,
            "docs": [],
            "answer": "",
            "messages": [HumanMessage(content=question)],
            "eval_mode": True,  # Activa LLM-as-Judge
        }
        config = {"configurable": {"thread_id": f"eval_{i}"}}

        try:
            final = await graph.ainvoke(inputs, config=config)
            answer = final.get("answer", "")[:200]

            # Obtener trazabilidad
            final_state = await graph.aget_state(config)
            traza = final_state.values.get("trazabilidad", {})
            metricas = traza.get("metricas", {})

            # Intent clasificado
            cls = traza.get("clasificacion", {})
            actual_intent = cls.get("intent", "?")

            # Metricas de retrieval
            ret = metricas.get("retrieval")
            if ret:
                retrieval_scores["recall"].append(ret["recall_at_k"])
                retrieval_scores["precision"].append(ret["precision_at_k"])
                retrieval_scores["mrr"].append(ret["mrr"])
                retrieval_scores["ndcg"].append(ret["ndcg_at_k"])

            # Metricas LLM judge
            judge = metricas.get("llm_judge")
            if judge:
                judge_scores["relevance"].append(judge["relevance_score"])
                judge_scores["faithfulness"].append(judge["faithfulness_score"])

            # Grounding score
            gs = metricas.get("grounding_score")
            if gs is not None:
                grounding_scores.append(gs)

            result = {
                "question": question,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_correct": actual_intent.lower().replace("ú", "u").replace("ó", "o")
                    == expected_intent.lower().replace("ú", "u").replace("ó", "o"),
                "retrieval_metrics": ret,
                "llm_judge_metrics": judge,
                "grounding_score": gs,
                "answer_preview": answer,
            }
            results.append(result)

            # Status line
            ret_str = f"R@k={ret['recall_at_k']:.2f}" if ret else "N/A"
            rel_str = f"Rel={judge['relevance_score']:.2f}" if judge else "N/A"
            faith_str = f"Faith={judge['faithfulness_score']:.2f}" if judge else "N/A"
            print(f"  Intent: {actual_intent} {'OK' if result['intent_correct'] else 'WRONG'} | {ret_str} | {rel_str} | {faith_str}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": question,
                "expected_intent": expected_intent,
                "error": str(e),
            })

    # ── Resumen ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN DE EVALUACION")
    print("=" * 70)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    intent_correct = sum(1 for r in results if r.get("intent_correct"))
    print(f"\nIntent Classification: {intent_correct}/{len(results)} correctos ({intent_correct/len(results)*100:.0f}%)")

    if retrieval_scores["recall"]:
        print(f"\nRetrieval Metrics (n={len(retrieval_scores['recall'])}):")
        print(f"  Recall@k:    {_avg(retrieval_scores['recall']):.4f}")
        print(f"  Precision@k: {_avg(retrieval_scores['precision']):.4f}")
        print(f"  MRR:         {_avg(retrieval_scores['mrr']):.4f}")
        print(f"  nDCG@k:      {_avg(retrieval_scores['ndcg']):.4f}")

    if judge_scores["relevance"]:
        print(f"\nLLM-as-Judge (n={len(judge_scores['relevance'])}):")
        print(f"  Relevance:    {_avg(judge_scores['relevance']):.4f}")
        print(f"  Faithfulness: {_avg(judge_scores['faithfulness']):.4f}")

    if grounding_scores:
        print(f"\nGrounding Score (n={len(grounding_scores)}): {_avg(grounding_scores):.4f}")

    # Guardar resultados
    output_path = Path("eval_results.json")
    summary = {
        "total_questions": len(EVAL_DATASET),
        "intent_accuracy": intent_correct / len(results) if results else 0,
        "avg_retrieval": {k: round(_avg(v), 4) for k, v in retrieval_scores.items()},
        "avg_llm_judge": {k: round(_avg(v), 4) for k, v in judge_scores.items()},
        "avg_grounding_score": round(_avg(grounding_scores), 4),
        "details": results,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResultados guardados en: {output_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
