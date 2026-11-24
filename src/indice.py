"""Índice invertido, frecuencias documentales y vocabulario."""
from collections import Counter
import math


def indice_inv(docs_procesados):
    """Asocia cada término con los documentos y sus frecuencias."""
    indice, longitudes, unicos = {}, {}, {}
    elementos = docs_procesados.items() if hasattr(docs_procesados, 'items') else docs_procesados
    for docid, tokens in elementos:
        frecuencias = Counter(tokens)
        longitudes[docid] = len(tokens)
        unicos[docid] = len(frecuencias)
        for termino, tf in frecuencias.items():
            indice.setdefault(termino, {})[docid] = tf
    return indice, longitudes, unicos


def docfrec(indice):
    """Cantidad de documentos que contienen cada término."""
    return {termino: len(postings) for termino, postings in indice.items()}


def get_voc(indice):
    """Conjunto de términos presentes en el índice."""
    return set(indice)


def construir(docs_procesados, metodo):
    indice, longitudes, unicos = indice_inv(docs_procesados)
    return finalizar(indice, longitudes, unicos, metodo)


def construir_varios(documentos_procesados, metodos):
    estructuras = {m: ({}, {}, {}) for m in metodos}
    for docid, variantes in documentos_procesados:
        for metodo, tokens in variantes.items():
            indice, longitudes, unicos = estructuras[metodo]
            frecuencias = Counter(tokens)
            longitudes[docid] = len(tokens)
            unicos[docid] = len(frecuencias)
            for termino, tf in frecuencias.items():
                indice.setdefault(termino, {})[docid] = tf
    return {m: finalizar(*estructuras[m], m) for m in metodos}


def finalizar(indice, longitudes, unicos, metodo):
    n = len(longitudes)
    if not n:
        raise ValueError('Corpus vacío')
    idf, normas2 = {}, dict.fromkeys(longitudes, 0.0)
    for termino, postings in indice.items():
        idf[termino] = math.log((n + 1) / (len(postings) + 1)) + 1
        for docid, tf in postings.items():
            normas2[docid] += (tf * idf[termino]) ** 2
    return {
        'metodo': metodo, 'N': n,
        'indice': indice, 'longitudes': longitudes, 'unicos': unicos,
        'idf': idf, 'normas': {d: math.sqrt(v) for d, v in normas2.items()},
        'promedio_longitud': sum(longitudes.values()) / n
    }
