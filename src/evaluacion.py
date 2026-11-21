"""Precision@k, Recall@k, AP completo y AP@k con denominador R total."""
from statistics import mean


def precision(recuperados, relevantes, k):
    """Fracción de las k posiciones solicitadas que son relevantes."""
    if k < 1:
        raise ValueError('k debe ser positivo')
    aciertos = 0
    for docid in recuperados[:k]:
        if docid in relevantes:
            aciertos += 1
    return aciertos / k


def recall(recuperados, relevantes, k):
    """Fracción de todos los relevantes encontrada en las primeras k posiciones."""
    if k < 1:
        raise ValueError('k debe ser positivo')
    if not relevantes:
        return 0.0
    aciertos = 0
    for docid in recuperados[:k]:
        if docid in relevantes:
            aciertos += 1
    return aciertos / len(relevantes)


def average_precision(recuperados, relevantes):
    """Promedio de precisión en las posiciones relevantes del ranking recibido."""
    if not relevantes:
        return 0.0
    aciertos, suma = 0, 0.0
    for posicion, docid in enumerate(recuperados, 1):
        if docid in relevantes:
            aciertos += 1
            suma += aciertos / posicion
    return suma / len(relevantes)


def mean_average_precision(valores_ap):
    if not valores_ap:
        raise ValueError('MAP requiere al menos una consulta')
    return sum(valores_ap) / len(valores_ap)


def metricas(ranking, qrels, k=10, completo=False):
    if k < 1:
        raise ValueError('k debe ser positivo')
    ids = [str(docid) for docid, _ in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError('Ranking con documentos repetidos')
    relevantes = {str(d) for d, score in qrels.items() if score > 0}
    recorrido = ids if completo else ids[:k]
    return {
        f'Precision@{k}': precision(ids, relevantes, k),
        f'Recall@{k}': recall(ids, relevantes, k),
        'AP' if completo else f'AP@{k}': average_precision(recorrido, relevantes)
    }


def resumir(filas, k=10, completo=True):
    if not filas:
        raise ValueError('No hay consultas evaluadas')
    ap = 'AP' if completo else f'AP@{k}'
    return {
        'consultas': len(filas),
        f'Precision@{k}': mean(f[f'Precision@{k}'] for f in filas),
        f'Recall@{k}': mean(f[f'Recall@{k}'] for f in filas),
        'MAP' if completo else f'MAP@{k}': mean_average_precision([f[ap] for f in filas]),
        'ms_por_consulta': mean(f['ms'] for f in filas)
    }
