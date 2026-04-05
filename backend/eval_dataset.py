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
    # ── Busqueda (datos puntuales) ──────────────────────────────────────────
    {
        "question": "Cual es la potencia del Toyota Hilux?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ft_toyota_hilux_v2_26"],
        "relevant_keywords": ["potencia", "hp", "cv", "kW"],
    },
    {
        "question": "Cuanto torque tiene el Mazda CX-5?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026"],
        "relevant_keywords": ["torque", "Nm"],
    },
    {
        "question": "Cual es el consumo de combustible del Corolla Cross?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ft_corolla_cross_v2_26"],
        "relevant_keywords": ["consumo", "km/l", "l/100"],
    },
    {
        "question": "Que tipo de transmision tiene el Volkswagen Tiguan?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["volkswagen_tiguan_elegance_my24_v1"],
        "relevant_keywords": ["transmision", "automatica", "manual", "DSG", "velocidades"],
    },
    {
        "question": "Cuales son las dimensiones del Peugeot 3008?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_nueva_3008_f_2_1"],
        "relevant_keywords": ["largo", "ancho", "alto", "mm", "distancia"],
    },
    {
        "question": "Que motor tiene el MG ZS Hybrid?",
        "expected_intent": "Busqueda",
        "relevant_doc_ids": ["ficha_tecnica_mg_zs_hybrid_mg"],
        "relevant_keywords": ["motor", "cilindros", "cc", "hibrido"],
    },
    # ── Resumen (ficha completa) ────────────────────────────────────────────
    {
        "question": "Dame un resumen de la ficha tecnica del Toyota Fortuner",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_fortuner_v_09_25"],
        "relevant_keywords": ["motor", "potencia", "transmision"],
    },
    {
        "question": "Resumeme las especificaciones del Mazda CX-30",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["motor", "dimensiones", "equipamiento"],
    },
    {
        "question": "Ficha completa del Opel Grandland",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["ficha_tecnica_opel_grandland_compressed"],
        "relevant_keywords": ["motor", "potencia", "seguridad"],
    },
    {
        "question": "Overview del Volkswagen Taos",
        "expected_intent": "Resumen",
        "relevant_doc_ids": ["volkswagen_taos_v8_my24_mex"],
        "relevant_keywords": ["motor", "equipamiento", "version"],
    },
    # ── Comparacion (2 modelos) ─────────────────────────────────────────────
    {
        "question": "Compara el Toyota Hilux vs el Volkswagen Amarok",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ft_toyota_hilux_v2_26", "12_11_1_2025_volkswagen_amarok_my26"],
        "relevant_keywords": ["potencia", "torque", "motor"],
    },
    {
        "question": "Diferencias entre el Mazda CX-5 y el Mazda CX-30",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ficha_tecnica_mazda_cx_5_2026", "ficha_tecnica_mazda_cx_30_202511"],
        "relevant_keywords": ["motor", "dimensiones", "potencia"],
    },
    {
        "question": "Compara el Corolla Cross con el Yaris Cross",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["ft_corolla_cross_v2_26", "ft_yariscross_v2_26"],
        "relevant_keywords": ["motor", "potencia", "consumo"],
    },
    {
        "question": "Volkswagen Polo vs Seat Ibiza",
        "expected_intent": "Comparacion",
        "relevant_doc_ids": ["volkswagen_polo_pa_my2025_v3", "seatibiza_compressed_1"],
        "relevant_keywords": ["motor", "potencia", "precio"],
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
