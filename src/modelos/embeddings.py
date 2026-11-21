"""Sentence Transformers + búsqueda exacta FAISS con coseno normalizado."""
import json
from pathlib import Path
import streamlit as st
from src.archivos import huella

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
    ids = list(documentos)
    vectores = encoder.encode(list(documentos.values()), batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True)
    vectores = np.ascontiguousarray(vectores, dtype='float32')
    sin_texto = [docid for docid in ids if not documentos[docid].strip()]
    for posicion, docid in enumerate(ids):
        if docid in sin_texto:
            vectores[posicion] = 0
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)
    faiss.write_index(indice, str(carpeta / 'vectores.faiss'))
    metadata = {'modelo': MODELO, 'revision': revision, 'ids': ids,
                'huella': huella(documentos), 'dimensiones': vectores.shape[1],
                'max_seq_length': 256, 'normalizados': True, 'sin_texto': sin_texto,
                'politica_vacios': 'vector_cero'}
    (carpeta / 'vectores.json').write_text(json.dumps(metadata), encoding='utf-8')


class ModeloEmbeddings:
    def __init__(self, carpeta, fingerprint):
        import faiss
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / 'vectores.json').read_text(encoding='utf-8'))
        if self.meta['huella'] != fingerprint:
            raise ValueError('Los embeddings pertenecen a un corpus diferente')
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
        scores, posiciones = self.indice.search(np.asarray(vector, dtype='float32'), self.indice.ntotal)
        ranking = [(self.meta['ids'][int(i)], float(s)) for s, i in zip(scores[0], posiciones[0]) if i >= 0]
        ranking.sort(key=lambda x: (-x[1], x[0]))
        return ranking if completo else [(d, s) for d, s in ranking if d not in self.sin_texto][:top_k]
