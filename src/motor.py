from pathlib import Path
from threading import Lock
from src.archivos import leer
from src.modelos.clasicos import ModelosClasicos
from src.preprocesamiento import procesar_texto


class Motor:
    """Carga los índices y dirige cada consulta al modelo seleccionado."""
    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)
        self.datos = leer(self.carpeta / 'corpus.json.gz')
        self.seleccion = leer(self.carpeta / 'seleccion.json.gz')
        self.metodo = self.seleccion['metodo']
        indice = leer(self.carpeta / f'indice_{self.metodo}.json.gz')
        self.clasicos = ModelosClasicos(indice)
        self.semantico = None
        self._carga_semantica = Lock()

    def buscar(self, texto, modelo, top_k=10, completo=False):
        """Devuelve documentos y puntajes ordenados; completo conserva todo el ranking."""
        if top_k < 1:
            raise ValueError('top_k debe ser positivo')
        if modelo == 'embeddings':
            with self._carga_semantica:
                if self.semantico is None:
                    from src.modelos.embeddings import ModeloEmbeddings
                    self.semantico = ModeloEmbeddings(self.carpeta)
            return self.semantico.buscar(texto, top_k, completo)
        tokens = procesar_texto(texto, self.metodo)
        return self.clasicos.buscar(tokens, modelo, top_k, completo)
