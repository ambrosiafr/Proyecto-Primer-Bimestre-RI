import argparse
from src.archivos import ROOT, guardar, leer
from src.datos import cargar_fiqa
from src.indice import construir_varios
from src.preprocesamiento import METODOS, procesar_documentos_varios


def main():
    p = argparse.ArgumentParser(description='Descargar FiQA y construir ambos índices clásicos')
    p.add_argument('--metodo', choices=METODOS, help='Preparar solo una alternativa (por defecto ambas)')
    p.add_argument('--workers', type=int, default=1, help='Procesos spaCy, aumentar si hay CPU/RAM disponibles')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers debe ser positivo')
    salida = ROOT / 'artefactos'
    cache = ROOT / 'data' / 'fiqa.json.gz'
    if cache.exists():
        datos = leer(cache)
        documentos, consultas, qrels = datos['documentos'], datos['consultas'], datos['qrels']
    else:
        documentos, consultas, qrels = cargar_fiqa(ROOT / 'data')
        guardar(cache, {'documentos': documentos, 'consultas': consultas, 'qrels': qrels})
    guardar(salida / 'corpus.json.gz', {
        'documentos': documentos, 'consultas': consultas, 'qrels': qrels,
        'completo': True,
        'origen': 'FiQA',
    })
    metodos = [args.metodo] if args.metodo else list(METODOS)
    print(f'Preparando {metodos}: {len(documentos)} documentos', flush=True)
    indices = construir_varios(procesar_documentos_varios(documentos, metodos, args.workers), metodos)
    for metodo, datos in indices.items():
        guardar(salida / f'indice_{metodo}.json.gz', datos)
        print(f'Vocabulario: {len(datos["indice"])}', flush=True)
    # Los artefactos dependen del corpus: se regeneran con el nuevo conjunto.
    for nombre in ('seleccion.json.gz', 'evaluacion_dev.json.gz', 'evaluacion_test.json.gz',
                   'vectores.faiss', 'vectores.json'):
        ruta = salida / nombre
        if ruta.exists():
            ruta.unlink()
    print('Índices guardados. Siguiente paso: comparar_preprocesamiento.', flush=True)


if __name__ == '__main__':
    main()
