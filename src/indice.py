"""Construcción del índice invertido y estadísticas de los documentos."""
from collections import Counter
import math


def construir_varios(documentos_procesados, metodos):
    estructuras = {}
    for metodo in metodos:
        estructuras[metodo] = {
            'indice': {},
            'longitudes': {},
            'unicos': {}
        }

    for docid, variantes in documentos_procesados:
        for metodo, tokens in variantes.items():
            datos = estructuras[metodo]
            frecuencias = Counter(tokens)
            datos['longitudes'][docid] = len(tokens)
            datos['unicos'][docid] = len(frecuencias)

            for termino, frecuencia in frecuencias.items():
                if termino not in datos['indice']:
                    datos['indice'][termino] = {}
                datos['indice'][termino][docid] = frecuencia

    indices = {}
    for metodo in metodos:
        datos = estructuras[metodo]
        indices[metodo] = finalizar(
            datos['indice'], datos['longitudes'], datos['unicos'], metodo
        )
    return indices


def finalizar(indice, longitudes, unicos, metodo):
    cantidad_documentos = len(longitudes)
    if cantidad_documentos == 0:
        raise ValueError('Corpus vacío')

    idf = {}
    sumas_cuadrados = {}
    for docid in longitudes:
        sumas_cuadrados[docid] = 0.0

    for termino, documentos in indice.items():
        frecuencia_documental = len(documentos)
        idf[termino] = math.log((cantidad_documentos + 1) / (frecuencia_documental + 1)) + 1
        for docid, frecuencia in documentos.items():
            peso = frecuencia * idf[termino]
            sumas_cuadrados[docid] += peso ** 2

    normas = {}
    for docid, suma in sumas_cuadrados.items():
        normas[docid] = math.sqrt(suma)

    return {
        'metodo': metodo,
        'N': cantidad_documentos,
        'indice': indice,
        'longitudes': longitudes,
        'unicos': unicos,
        'idf': idf,
        'normas': normas,
        'promedio_longitud': sum(longitudes.values()) / cantidad_documentos
    }
