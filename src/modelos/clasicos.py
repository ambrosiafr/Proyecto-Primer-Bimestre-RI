"""Modelos propios. Se recorren postings, sin matrices densas ni motores externos."""
from collections import Counter
import math

MODELOS = ('jaccard', 'tfidf', 'bm25')


class ModelosClasicos:
    def __init__(self, datos, k1=1.2, b=0.75):
        self.datos = datos
        self.indice = datos['indice']
        self.ids = sorted(datos['longitudes'])
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Parámetros BM25 inválidos')
        self.k1 = k1
        self.b = b

    def buscar(self, tokens, modelo='bm25', top_k=10, completo=False):
        if modelo not in MODELOS:
            raise ValueError('Modelo desconocido')
        if top_k < 1:
            raise ValueError('top_k debe ser positivo')
        if modelo == 'jaccard':
            scores = self.jaccard(tokens)
        elif modelo == 'tfidf':
            scores = self.tfidf(tokens)
        else:
            scores = self.bm25(tokens)
        # Para MAP completo se incluyen también documentos con score cero.
        if completo:
            for docid in self.ids:
                if docid not in scores:
                    scores[docid] = 0.0
        # Puntaje de mayor a menor; en empate, ID de menor a mayor.
        ranking = sorted(scores.items(), key=lambda par: (-par[1], par[0]))
        if completo:
            return ranking
        return ranking[:top_k]

    def jaccard(self, tokens):
        """Intersección entre términos únicos dividida por su unión."""
        query = set(tokens)  # Incluye OOV en la unión; representación binaria.
        intersecciones = {}
        for termino in query:
            for docid in self.indice.get(termino, {}):
                if docid not in intersecciones:
                    intersecciones[docid] = 0
                intersecciones[docid] += 1
        scores = {}
        for docid, comunes in intersecciones.items():
            union = len(query) + self.datos['unicos'][docid] - comunes
            if union > 0:
                scores[docid] = comunes / union
            else:
                scores[docid] = 0.0
        return scores

    def tfidf(self, tokens):
        """Coseno entre los pesos TF-IDF de consulta y documento."""
        query = {}
        for termino, tf in Counter(tokens).items():
            if termino in self.indice:
                query[termino] = tf * self.datos['idf'][termino]
        suma_cuadrados = 0.0
        for peso in query.values():
            suma_cuadrados += peso * peso
        norma = math.sqrt(suma_cuadrados)
        scores = {}
        if not norma:
            return scores
        for termino, peso in query.items():
            for docid, tf in self.indice[termino].items():
                peso_doc = tf * self.datos['idf'][termino]
                if docid not in scores:
                    scores[docid] = 0.0
                scores[docid] += peso * peso_doc
        for docid in scores:
            scores[docid] /= norma * self.datos['normas'][docid]
        return scores

    def bm25(self, tokens):
        """Puntaje por rareza del término, frecuencia y longitud documental."""
        scores = {}
        longitud_promedio = self.datos['promedio_longitud']
        if not longitud_promedio:
            return scores
        for termino in sorted(set(tokens)):
            postings = self.indice.get(termino, {})
            df = len(postings)
            idf = math.log(1 + (self.datos['N'] - df + 0.5) / (df + 0.5))
            for docid, tf in postings.items():
                longitud = self.datos['longitudes'][docid]
                denominador = tf + self.k1 * (1 - self.b + self.b * longitud / longitud_promedio)
                if docid not in scores:
                    scores[docid] = 0.0
                scores[docid] += idf * tf * (self.k1 + 1) / denominador
        return scores
