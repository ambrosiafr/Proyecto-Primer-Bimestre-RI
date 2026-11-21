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
        if self.seleccion.get('version_modelos') != 2:
            raise ValueError('La selección debe repetirse con las fórmulas actuales')
        self.metodo = self.seleccion['metodo']
        indice = leer(self.carpeta / f'indice_{self.metodo}.json.gz')
        if indice.get('version') != 2:
            raise ValueError('El índice requiere actualizarse para las nuevas fórmulas')
        if indice['huella'] != self.datos['huella']:
            raise ValueError('Índice incompatible con el corpus')
        if set(indice['longitudes']) != set(self.datos['documentos']):
            raise ValueError('El índice clásico no cubre todos los documentos del corpus')
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
                    self.semantico = ModeloEmbeddings(self.carpeta, self.datos['huella'])
            return self.semantico.buscar(texto, top_k, completo)
        tokens = procesar_texto(texto, self.metodo)
        return self.clasicos.buscar(tokens, modelo, top_k, completo)
