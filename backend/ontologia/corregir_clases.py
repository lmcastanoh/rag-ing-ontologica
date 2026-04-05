# backend/ontologia/corregir_clases.py
# Corrige clasificaciones erróneas en vehiculos_completo.ttl sin rellamar al LLM.

import re
from pathlib import Path

TTL = Path(__file__).parent / "vehiculos_completo.ttl"

content = TTL.read_text(encoding="utf-8")
original = content  # para comparar al final

# ==============================================================================
# UTILIDADES
# ==============================================================================

def cambiar_clase_vehiculo(ttl: str, uri_id: str, clase_vieja: str, clase_nueva: str) -> str:
    """Cambia la clase rdf:type de un individuo vehículo."""
    return ttl.replace(
        f":{uri_id} a :{clase_vieja} ;",
        f":{uri_id} a :{clase_nueva} ;"
    )


def combustion_a_electrico(ttl: str, uri_id: str) -> str:
    """
    Convierte un vehículo y su motor de Combustión -> Eléctrico:
    - Cambia clase del vehículo
    - Renombra motor _Comb -> _Elec y cambia su clase
    - Cambia tieneMotorCombustion -> tieneMotorElectrico
    - Elimina usaCombustible y tieneCilindradaCc del motor
    - Elimina tieneConsumoL100km del vehículo
    """
    motor_comb = f":Motor_{uri_id}_Comb"
    motor_elec = f":Motor_{uri_id}_Elec"

    # 1. Clase del vehículo
    ttl = ttl.replace(
        f":{uri_id} a :VehiculoCombustion ;",
        f":{uri_id} a :VehiculoElectrico ;"
    )

    # 2. Clase del motor
    ttl = ttl.replace(
        f"{motor_comb} a :MotorCombustion ;",
        f"{motor_elec} a :MotorElectrico ;"
    )

    # 3. Todas las referencias al motor en el vehículo
    ttl = ttl.replace(
        f":tieneMotorCombustion    {motor_comb}",
        f":tieneMotorElectrico     {motor_elec}"
    )

    # 4. Renombrar el URI del motor en el resto del archivo
    ttl = ttl.replace(motor_comb, motor_elec)

    # 5. Eliminar líneas de combustible (solo del bloque de este motor)
    #    Buscamos el bloque del motor y limpiamos líneas específicas
    def limpiar_motor(m):
        bloque = m.group(0)
        bloque = re.sub(r"\s+:usaCombustible\s+:\w+ ;?\n?", "\n", bloque)
        bloque = re.sub(r"\s+:tieneCilindradaCc\s+\d+ ;?\n?", "\n", bloque)
        # Si el último campo terminó con ";" y ahora quedó solo, cambiarlo a "."
        bloque = re.sub(r":tienePotenciaCV\s+(\d+)\s+;(\s*\n\s*\.)", r":tienePotenciaCV \1 .", bloque)
        return bloque

    patron_motor = re.compile(
        rf"{re.escape(motor_elec)} a :MotorElectrico ;.*?(?=\n\n|\Z)",
        re.DOTALL
    )
    ttl = patron_motor.sub(limpiar_motor, ttl)

    # 6. Eliminar tieneConsumoL100km del vehículo
    ttl = re.sub(
        rf"(\s+:tieneConsumoL100km\s+[\d\.]+\s+;?\n)",
        lambda m: "" if uri_id in ttl[max(0, ttl.find(m.group(0))-500):ttl.find(m.group(0))+10] else m.group(0),
        ttl
    )

    return ttl


def hibrido_a_combustion(ttl: str, uri_id: str) -> str:
    """
    Convierte un vehículo de Híbrido -> Combustión:
    - Cambia la clase del vehículo
    - Elimina el motor eléctrico si existe
    - Cambia tieneMotorElectrico -> tieneMotorCombustion (si aplica)
    """
    motor_elec = f":Motor_{uri_id}_Elec"
    motor_comb = f":Motor_{uri_id}_Comb"

    # Cambiar clase
    ttl = ttl.replace(
        f":{uri_id} a :VehiculoHibrido ;",
        f":{uri_id} a :VehiculoCombustion ;"
    )

    # Si tiene motor eléctrico, eliminarlo del vehículo y cambiar propiedad
    ttl = re.sub(
        rf"\s+:tieneMotorElectrico\s+{re.escape(motor_elec)}\s+;?\n?",
        "\n",
        ttl
    )

    # Eliminar el bloque del motor eléctrico si existe
    ttl = re.sub(
        rf"{re.escape(motor_elec)} a :MotorElectrico ;.*?\.\n\n",
        "",
        ttl,
        flags=re.DOTALL
    )

    return ttl


def combustion_a_hibrido(ttl: str, uri_id: str, potencia_elec: int = 80) -> str:
    """
    Convierte un vehículo de Combustión -> Híbrido:
    - Cambia la clase del vehículo
    - Agrega motor eléctrico
    - Agrega tieneMotorElectrico
    """
    motor_elec = f":Motor_{uri_id}_Elec"
    motor_comb = f":Motor_{uri_id}_Comb"

    # Cambiar clase
    ttl = ttl.replace(
        f":{uri_id} a :VehiculoCombustion ;",
        f":{uri_id} a :VehiculoHibrido ;"
    )

    # Agregar motor eléctrico antes del bloque del vehículo
    bloque_motor_elec = (
        f"{motor_elec} a :MotorElectrico ;\n"
        f"    :tienePotenciaCV {potencia_elec} .\n\n"
    )

    marcador = f":{uri_id} a :VehiculoHibrido ;"
    ttl = ttl.replace(marcador, bloque_motor_elec + marcador)

    # Agregar tieneMotorElectrico al vehículo (después de tieneMotorCombustion)
    ttl = ttl.replace(
        f":tieneMotorCombustion    {motor_comb} ;",
        f":tieneMotorCombustion    {motor_comb} ;\n    :tieneMotorElectrico     {motor_elec} ;"
    )

    return ttl


# ==============================================================================
# CORRECCIONES
# ==============================================================================

print("Aplicando correcciones al TTL...")

# --- MG Emotor eléctricos mal clasificados como combustión ---
mg_electricos = [
    "MgEmotor_Cybester",
    "MgEmotor_Marvel",
    "MgEmotor_MgRx5",
    "MgEmotor_MgS5",
    "MgEmotor_Zs",
]
for uri in mg_electricos:
    antes = content.count(f":{uri} a :VehiculoElectrico")
    content = combustion_a_electrico(content, uri)
    despues = content.count(f":{uri} a :VehiculoElectrico")
    print(f"  [OK] {uri}: VehiculoCombustion -> VehiculoElectrico ({antes}->{despues})")

# --- Mazda M3 y CX-30 mal clasificados como híbridos ---
mazda_combustion = [
    "Mazda_M3",
    "Mazda_MazdaCx30",
]
for uri in mazda_combustion:
    antes = content.count(f":{uri} a :VehiculoHibrido")
    content = hibrido_a_combustion(content, uri)
    despues = content.count(f":{uri} a :VehiculoCombustion")
    print(f"  [OK] {uri}: VehiculoHibrido -> VehiculoCombustion ({antes}->{despues})")

# --- Toyota Corolla Cross: combustión -> híbrido ---
content = combustion_a_hibrido(content, "Toyota_CorollaCross", potencia_elec=88)
print(f"  [OK] Toyota_CorollaCross: VehiculoCombustion -> VehiculoHibrido")

# ==============================================================================
# VERIFICACIÓN Y GUARDADO
# ==============================================================================

# Contar clases finales
v_comb  = len(re.findall(r'a :VehiculoCombustion ;', content))
v_elec  = len(re.findall(r'a :VehiculoElectrico ;', content))
v_hibr  = len(re.findall(r'a :VehiculoHibrido ;', content))
m_comb  = len(re.findall(r'a :MotorCombustion ;', content))
m_elec  = len(re.findall(r'a :MotorElectrico ;', content))

print(f"\nResultado final en el TTL:")
print(f"  VehiculoCombustion : {v_comb}")
print(f"  VehiculoElectrico  : {v_elec}")
print(f"  VehiculoHibrido    : {v_hibr}")
print(f"  Total vehículos    : {v_comb + v_elec + v_hibr}")
print(f"  MotorCombustion    : {m_comb}")
print(f"  MotorElectrico     : {m_elec}")

# Verificar que no se rompió el TTL (sin puntos en URIs)
uris_con_punto = re.findall(r':[A-Za-z_][A-Za-z0-9_]*\.(?=\s)', content)
if uris_con_punto:
    print(f"\n  ADVERTENCIA: URIs con punto encontrados: {uris_con_punto}")
else:
    print(f"\n  Sin URIs con punto problematico - OK")

TTL.write_text(content, encoding="utf-8")
print(f"\nArchivo guardado: {TTL}")
print("Siguiente paso: reimportar vehiculos_completo.ttl en GraphDB con 'Clear repository'")
