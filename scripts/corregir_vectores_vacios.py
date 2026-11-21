"""Actualiza índices existentes sin volver a codificar los documentos con texto."""
import json
from src.archivos import ROOT, leer


def main():
    import faiss
    carpeta = ROOT / 'artefactos'
    corpus = leer(carpeta / 'corpus.json.gz')
    ruta = carpeta / 'vectores.json'
    meta = json.loads(ruta.read_text(encoding='utf-8'))
    if meta['huella'] != corpus['huella']:
        raise ValueError('Corpus e índice vectorial incompatibles')
    indice = faiss.read_index(str(carpeta / 'vectores.faiss'))
    vectores = indice.reconstruct_n(0, indice.ntotal)
    sin_texto = []
    for posicion, docid in enumerate(meta['ids']):
        if not corpus['documentos'][docid].strip():
            vectores[posicion] = 0
            sin_texto.append(docid)
    nuevo = faiss.IndexFlatIP(indice.d)
    nuevo.add(vectores)
    faiss.write_index(nuevo, str(carpeta / 'vectores.faiss'))
    meta.update(sin_texto=sin_texto, politica_vacios='vector_cero')
    ruta.write_text(json.dumps(meta), encoding='utf-8')
    print(f'{len(sin_texto)} vectores vacíos corregidos; {nuevo.ntotal} documentos conservados.')


if __name__ == '__main__':
    main()
