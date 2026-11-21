"""Agrupación descriptiva, sin entrenar ni ajustar modelos con test."""
import re
from statistics import mean


def tipo_consulta(texto):
    """Reglas explícitas; no detectan automáticamente paráfrasis o intención."""
    tokens = re.findall(r'[a-z0-9]+', texto.lower())
    if set(tokens) & {'tax', 'taxes', 'taxable', 'retirement', 'ira', '401'}:
        return 'Impuestos y jubilación'
    if set(tokens) & {'stock', 'stocks', 'investment', 'investing', 'fund', 'funds'}:
        return 'Inversión y mercados'
    if len(tokens) <= 6:
        return 'Otras consultas cortas (hasta 6 palabras)'
    return 'Otras consultas largas (>6 palabras)'


def por_tipo(reporte):
    grupos = {}
    for modelo, datos in reporte['modelos'].items():
        for fila in datos['consultas']:
            categoria = tipo_consulta(fila['consulta'])
            grupos.setdefault((categoria, modelo), []).append(fila['AP'])
    return [{'tipo': categoria, 'modelo': modelo, 'consultas': len(aps), 'MAP': mean(aps)}
            for (categoria, modelo), aps in sorted(grupos.items())]
