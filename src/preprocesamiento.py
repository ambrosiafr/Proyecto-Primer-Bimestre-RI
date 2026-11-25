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
        tiene_letra_o_numero = False
        for caracter in texto:
            if caracter.isalnum():
                tiene_letra_o_numero = True
                break
        if not tiene_letra_o_numero and texto not in NEGACIONES and texto not in SIMBOLOS:
            continue
        if metodo == 'stemming':
            termino = raiz(texto)
        else:
            termino = token.lemma_
            if not termino:
                termino = token.text
            termino = termino.lower()
        if termino:
            resultado.append(termino)
    return resultado


def procesar_texto(texto, metodo):
    texto = texto.replace('’', "'")
    nlp = cargar_nlp()
    if metodo == 'stemming':
        doc = nlp.make_doc(texto)
    else:
        doc = nlp(texto)
    return transformar_doc(doc, metodo)


def recorrer_textos(documentos):
    """Entrega los textos con apóstrofos uniformes sin copiar todo el corpus."""
    for texto in documentos.values():
        yield texto.replace('’', "'")


def procesar_documentos_varios(documentos, metodos, workers=1):
    if not metodos:
        raise ValueError('Faltan los métodos de preprocesamiento')
    for metodo in metodos:
        if metodo not in METODOS:
            raise ValueError('Método de preprocesamiento inválido')

    desactivar = []
    if metodos == ['stemming']:
        desactivar = ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer']

    nlp = cargar_nlp()
    textos = recorrer_textos(documentos)
    documentos_spacy = nlp.pipe(textos, batch_size=128, n_process=workers, disable=desactivar)

    posicion = 0
    # El diccionario y pipe conservan el mismo orden de documentos.
    for docid, doc in zip(documentos, documentos_spacy):
        posicion += 1
        variantes = {}
        for metodo in metodos:
            variantes[metodo] = transformar_doc(doc, metodo)
        if len(documentos) > 1000 and posicion % 5000 == 0:
            print(f'{posicion}/{len(documentos)} documentos procesados', flush=True)
        yield docid, variantes
