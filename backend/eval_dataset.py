# backend/eval_dataset.py
# ==============================================================================
# Dataset de ground truth para evaluacion del sistema RAG.
#
# Cada entrada contiene:
#   - question:          Pregunta de evaluacion
#   - expected_intent:   Intent esperado (Busqueda|Resumen|Comparacion|GENERAL)
#   - relevant_doc_ids:  Doc IDs relevantes en ChromaDB (ground truth para retrieval)
#   - relevant_keywords: Palabras clave esperadas en la respuesta
# ==============================================================================
from __future__ import annotations

EVAL_DATASET: list[dict] = [
    # ── Busqueda (datos puntuales) - modelos con datos verificados en PDFs ──
    {
        "question": "Cual es la potencia del Mazda CX-30?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["potencia", "hp", "153"],
    },
    {
        "question": "Que torque tiene el Mazda CX-30?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["torque", "kg-m", "20.39"],
    },
    {
        "question": "Que motores tiene el Mazda CX-5?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026"],
        "relevant_keywords": ["motor", "Skyactiv-G", "2.0L", "2.5L"],
    },
    {
        "question": "Que transmision tiene el Mazda CX-5?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026"],
        "relevant_keywords": ["transmision", "Skyactiv-Drive", "6", "velocidades"],
    },
    {
        "question": "Cual es la potencia del Mazda 3?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_m3_202511"],
        "relevant_keywords": ["potencia", "hp", "153"],
    },
    {
        "question": "Que motor tiene el Mazda MX-5?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_mx_5_2026"],
        "relevant_keywords": ["motor", "Skyactiv"],
    },
    # ── Resumen (ficha completa) - modelos con buena cobertura ──────────────
    {
        "question": "Resumeme las especificaciones del Mazda CX-30",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["motor", "dimensiones", "equipamiento"],
    },
    {
        "question": "Dame un resumen del Mazda CX-5",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026"],
        "relevant_keywords": ["motor", "transmision", "Touring"],
    },
    {
        "question": "Ficha completa del Mazda 3",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_m3_202511"],
        "relevant_keywords": ["motor", "MHEV", "potencia"],
    },
    {
        "question": "Overview del Opel Grandland",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_opel_grandland_compressed"],
        "relevant_keywords": ["motor", "garantia"],
    },
    # ── Comparacion (2 modelos) - ambos con datos verificados ───────────────
    {
        "question": "Diferencias entre el Mazda CX-5 y el Mazda CX-30",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026", "ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["motor", "Skyactiv", "dimensiones"],
    },
    {
        "question": "Compara el Mazda 3 vs el Mazda CX-30",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ficha_tecnica_m3_202511", "ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["motor", "potencia", "torque"],
    },
    {
        "question": "Compara el Mazda CX-5 con el Mazda MX-5",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026", "ficha_tecnica_mazda_mx_5_2026"],
        "relevant_keywords": ["motor", "Skyactiv"],
    },
    {
        "question": "Opel Grandland vs Opel Mokka",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ficha_tecnica_opel_grandland_compressed", "opel_mokka_2023_2_compressed"],
        "relevant_keywords": ["motor", "potencia"],
    },
    # ── GENERAL (sin RAG) ──────────────────────────────────────────────────
    {
        "question": "Que es un motor turbo?",
        "expected_intent": "GENERAL",
        "relevant_doc_ids": [],
        "relevant_keywords": ["turbo", "compresor", "aire", "potencia"],
    },
    {
        "question": "Cual es la diferencia entre traccion 4x4 y AWD?",
        "expected_intent": "GENERAL",
        "relevant_doc_ids": [],
        "relevant_keywords": ["traccion", "4x4", "AWD", "ruedas"],
    },
    {
        "question": "Que significa CVT en una transmision?",
        "expected_intent": "GENERAL",
        "relevant_doc_ids": [],
        "relevant_keywords": ["CVT", "continuamente", "variable", "transmision"],
    },
]
