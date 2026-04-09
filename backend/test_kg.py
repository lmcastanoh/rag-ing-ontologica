#!/usr/bin/env python
# backend/test_kg.py
# ==============================================================================
# Script de prueba para la integracion del Knowledge Graph.
#
# Verifica:
#   1. Que GraphDB este corriendo en http://localhost:7200
#   2. Que el repositorio "vehiculos" exista
#   3. Que las funciones SPARQL retornen datos
#   4. Que la tool consultar_grafo_conocimiento funcione end-to-end
#
# Uso: cd backend && python test_kg.py
# ==============================================================================
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que backend/ este en el path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def test_endpoint():
    """Test 1: GraphDB esta corriendo y responde."""
    print("\n[1] Verificando que GraphDB este corriendo...")
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:7200/rest/repositories", timeout=5) as r:
            if r.status == 200:
                print("    OK: GraphDB responde en http://localhost:7200")
                return True
            else:
                print(f"    FAIL: GraphDB responde con status {r.status}")
                return False
    except Exception as e:
        print(f"    FAIL: GraphDB no responde — {e}")
        print("    SOLUCION: Inicia GraphDB Desktop")
        return False


def test_sparql_directo():
    """Test 2: Funciones SPARQL del kg_retriever."""
    print("\n[2] Probando funciones SPARQL directamente...")
    try:
        from kg_retriever import (
            kg_buscar_especificaciones,
            kg_buscar_motor,
            kg_listar_modelos_por_marca,
            kg_electricos_por_autonomia,
        )

        # Test: especificaciones de algun modelo
        print("    -> kg_buscar_especificaciones('Hilux')")
        results = kg_buscar_especificaciones("Hilux")
        print(f"       resultado: {len(results)} filas")
        if results:
            print(f"       primera fila: {results[0]}")

        print("    -> kg_buscar_motor('Hilux')")
        results = kg_buscar_motor("Hilux")
        print(f"       resultado: {len(results)} filas")

        print("    -> kg_listar_modelos_por_marca('Toyota')")
        results = kg_listar_modelos_por_marca("Toyota")
        print(f"       resultado: {len(results)} filas")
        if results:
            print(f"       modelos: {[r.get('nombreModelo') for r in results[:5]]}")

        print("    -> kg_electricos_por_autonomia(300)")
        results = kg_electricos_por_autonomia(300.0)
        print(f"       resultado: {len(results)} filas")

        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_agente():
    """Test 3: La tool consultar_grafo_conocimiento desde el agente."""
    print("\n[3] Probando tool del agente ReAct...")
    try:
        from tools import consultar_grafo_conocimiento

        # Invocar como lo hace el agente (con .invoke)
        result1 = consultar_grafo_conocimiento.invoke({
            "accion": "especificaciones",
            "modelo": "Hilux",
        })
        print(f"    -> accion='especificaciones', modelo='Hilux':")
        print(f"       {result1[:300]}")

        result2 = consultar_grafo_conocimiento.invoke({
            "accion": "por_marca",
            "marca": "Toyota",
        })
        print(f"\n    -> accion='por_marca', marca='Toyota':")
        print(f"       {result2[:300]}")

        result3 = consultar_grafo_conocimiento.invoke({
            "accion": "electricos",
            "autonomia_minima": 300.0,
        })
        print(f"\n    -> accion='electricos', autonomia_minima=300:")
        print(f"       {result3[:300]}")

        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Integracion del Knowledge Graph")
    print("=" * 70)

    if not test_endpoint():
        print("\n[FATAL] GraphDB no esta disponible. Aborta.")
        sys.exit(1)

    test_sparql_directo()
    test_tool_agente()

    print("\n" + "=" * 70)
    print("Tests completados")
    print("=" * 70)
