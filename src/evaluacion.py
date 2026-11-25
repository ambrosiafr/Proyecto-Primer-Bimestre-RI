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
    aciertos = 0
    suma = 0.0
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
    ids = []
    for docid, puntaje in ranking:
        ids.append(str(docid))
    if len(ids) != len(set(ids)):
        raise ValueError('Ranking con documentos repetidos')
    relevantes = set()
    for docid, relevancia in qrels.items():
        if relevancia > 0:
            relevantes.add(str(docid))
    if completo:
        recorrido = ids
        nombre_ap = 'AP'
    else:
        recorrido = ids[:k]
        nombre_ap = f'AP@{k}'
    return {
        f'Precision@{k}': precision(ids, relevantes, k),
        f'Recall@{k}': recall(ids, relevantes, k),
        nombre_ap: average_precision(recorrido, relevantes)
    }


def resumir(filas, k=10, completo=True):
    if not filas:
        raise ValueError('No hay consultas evaluadas')
    if completo:
        ap = 'AP'
        nombre_map = 'MAP'
    else:
        ap = f'AP@{k}'
        nombre_map = f'MAP@{k}'
    precisiones = []
    recalls = []
    valores_ap = []
    tiempos = []
    for fila in filas:
        precisiones.append(fila[f'Precision@{k}'])
        recalls.append(fila[f'Recall@{k}'])
        valores_ap.append(fila[ap])
        tiempos.append(fila['ms'])
    return {
        'consultas': len(filas),
        f'Precision@{k}': mean(precisiones),
        f'Recall@{k}': mean(recalls),
        nombre_map: mean_average_precision(valores_ap),
        'ms_por_consulta': mean(tiempos)
    }
