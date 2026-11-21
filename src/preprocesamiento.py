"""Mismo tokenizador y filtro en ambos experimentos; cambia solo la morfología."""
from functools import lru_cache

NEGACIONES = {'no', 'not', 'nor', "n't", 'never'}
SIMBOLOS = {'%', '$', '€', '£', '+', '-'}
METODOS = ('stemming', 'lematizacion')


@lru_cache(maxsize=1)
def cargar_nlp():
    import spacy
    return spacy.load('en_core_web_sm', disable=['parser', 'ner'])


@lru_cache(maxsize=1)
def cargar_stemmer():
    from nltk.stem import SnowballStemmer
    return SnowballStemmer('english')


@lru_cache(maxsize=100000)
def raiz(texto):
    return cargar_stemmer().stem(texto)


def transformar_doc(doc, metodo):
    if metodo not in METODOS:
        raise ValueError('Solo se permite stemming o lematizacion')
    resultado = []
    for token in doc:
        texto = token.lower_
        if token.is_space or (token.is_stop and texto not in NEGACIONES):
            continue
        if not (any(c.isalnum() for c in texto) or texto in NEGACIONES or texto in SIMBOLOS):
            continue
        if metodo == 'stemming':
            termino = raiz(texto)
        else:
            termino = (token.lemma_ or token.text).lower()
        if termino:
            resultado.append(termino)
    return resultado


def procesar_texto(texto, metodo):
    texto = texto.replace('’', "'")
    nlp = cargar_nlp()
    doc = nlp.make_doc(texto) if metodo == 'stemming' else nlp(texto)
    return transformar_doc(doc, metodo)


def procesar_documentos(documentos, metodo, workers=1):
    for docid, variantes in procesar_documentos_varios(documentos, [metodo], workers):
        yield docid, variantes[metodo]


def procesar_documentos_varios(documentos, metodos, workers=1):
    if not metodos or any(m not in METODOS for m in metodos):
        raise ValueError('Métodos de preprocesamiento inválidos')
    textos = (texto.replace('’', "'") for texto in documentos.values())
    desactivar = ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'] if metodos == ['stemming'] else []
    for posicion, (docid, doc) in enumerate(zip(documentos, cargar_nlp().pipe(
            textos, batch_size=128, n_process=workers, disable=desactivar)), 1):
        if len(documentos) > 1000 and posicion % 5000 == 0:
            print(f'{"/".join(metodos)}: {posicion}/{len(documentos)} documentos', flush=True)
        yield docid, {metodo: transformar_doc(doc, metodo) for metodo in metodos}
