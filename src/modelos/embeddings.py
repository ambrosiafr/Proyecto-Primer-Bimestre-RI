"""Sentence Transformers + búsqueda exacta FAISS con coseno normalizado."""
import json
from pathlib import Path
import streamlit as st

MODELO = 'sentence-transformers/all-MiniLM-L6-v2'


@st.cache_resource(show_spinner=False)
def codificador(modelo=MODELO, revision=None):
    """Codificador compartido por modelo y revisión entre búsquedas y enlaces."""
    import torch
    torch.set_num_threads(4)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sentence_transformers import SentenceTransformer
    try:
        carpeta = snapshot_download(modelo, revision=revision, local_files_only=True)
        necesarios = ('modules.json', 'config.json', 'model.safetensors',
                      'tokenizer.json', '1_Pooling/config.json')
        if not all((Path(carpeta) / f).is_file() for f in necesarios):
            raise LocalEntryNotFoundError('La copia local está incompleta')
    except LocalEntryNotFoundError:
        carpeta = snapshot_download(modelo, revision=revision,
                                    allow_patterns=['*.json', '*.safetensors', 'vocab.txt'])
    # Una ruta local evita peticiones al Hub por archivos opcionales en cada arranque.
    for intento in range(2):
        try:
            encoder = SentenceTransformer(carpeta, device='cpu', local_files_only=True)
            break
        except NotImplementedError as error:
            # El reintento utiliza los mismos pesos del checkpoint local.
            # to_empty dejaría parámetros sin los pesos preentrenados.
            if intento or 'meta tensor' not in str(error):
                raise
    if any(p.is_meta for p in (*encoder.parameters(), *encoder.buffers())):
        raise RuntimeError('El modelo no cargó sus pesos. Reinicia la aplicación para repetir la carga.')
    return encoder


def preparar(documentos, carpeta, batch_size=64):
    import faiss
    import numpy as np
    from huggingface_hub import HfApi
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(MODELO).sha
    encoder = codificador(MODELO, revision)
    encoder.max_seq_length = 256
    # La fila de cada vector corresponde al ID en esta misma posición.
    ids = list(documentos)
    textos = list(documentos.values())
    vectores = encoder.encode(textos, batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True)
    vectores = np.ascontiguousarray(vectores, dtype='float32')
    sin_texto = []
    for posicion, docid in enumerate(ids):
        if not documentos[docid].strip():
            sin_texto.append(docid)
            vectores[posicion] = 0
    dimensiones = vectores.shape[1]
    indice = faiss.IndexFlatIP(dimensiones)
    # FAISS recibe la matriz numérica; los textos permanecen en el corpus.
    indice.add(vectores)
    faiss.write_index(indice, str(carpeta / 'vectores.faiss'))
    metadata = {'modelo': MODELO, 'revision': revision, 'ids': ids,
                'dimensiones': dimensiones,
                'max_seq_length': 256, 'normalizados': True, 'sin_texto': sin_texto,
                'politica_vacios': 'vector_cero'}
    (carpeta / 'vectores.json').write_text(json.dumps(metadata), encoding='utf-8')


class ModeloEmbeddings:
    def __init__(self, carpeta):
        import faiss
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / 'vectores.json').read_text(encoding='utf-8'))
        self.indice = faiss.read_index(str(carpeta / 'vectores.faiss'))
        if self.indice.ntotal != len(self.meta['ids']) or self.indice.d != self.meta['dimensiones']:
            raise ValueError('Índice FAISS y metadatos incompatibles')
        self.encoder = codificador(self.meta['modelo'], self.meta['revision'])
        self.encoder.max_seq_length = self.meta['max_seq_length']
        self.sin_texto = set(self.meta.get('sin_texto', []))

    def buscar(self, texto, top_k=10, completo=False):
        import numpy as np
        if top_k < 1:
            raise ValueError('top_k debe ser positivo')
        if not texto.strip():
            return []
        vector = self.encoder.encode([texto], normalize_embeddings=True)
        # Recuperar todos permite desempates deterministas también en el corte k.
        vector = np.asarray(vector, dtype='float32')
        scores, posiciones = self.indice.search(vector, self.indice.ntotal)
        ranking = []
        for posicion, puntaje in zip(posiciones[0], scores[0]):
            if posicion >= 0:
                docid = self.meta['ids'][int(posicion)]
                ranking.append((docid, float(puntaje)))
        ranking.sort(key=lambda resultado: (-resultado[1], resultado[0]))
        if completo:
            return ranking
        resultados = []
        for docid, puntaje in ranking:
            if docid not in self.sin_texto:
                resultados.append((docid, puntaje))
            if len(resultados) == top_k:
                break
        return resultados
