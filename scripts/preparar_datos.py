import argparse
from src.archivos import ROOT, guardar, leer, huella
from src.datos import cargar_fiqa, leer_textos
from src.indice import construir_varios
from src.preprocesamiento import METODOS, procesar_documentos_varios


def main():
    p = argparse.ArgumentParser(description='Descargar FiQA y construir ambos índices clásicos')
    p.add_argument('--limite', type=int, help='Solo desarrollo; nunca evaluación oficial')
    p.add_argument('--carpeta-txt', help='Corpus propio UTF-8; sin qrels FiQA')
    p.add_argument('--metodo', choices=METODOS, help='Preparar solo una alternativa (por defecto ambas)')
    p.add_argument('--workers', type=int, default=1, help='Procesos spaCy, aumentar si hay CPU/RAM disponibles')
    args = p.parse_args()
    if args.limite is not None and args.limite < 1:
        p.error('--limite debe ser positivo')
    if args.workers < 1:
        p.error('--workers debe ser positivo')
    salida = ROOT / 'artefactos'
    cache = ROOT / 'data' / 'fiqa.json.gz'
    if args.carpeta_txt:
        documentos, consultas, qrels = leer_textos(args.carpeta_txt), {}, {}
    elif cache.exists():
        datos = leer(cache)
        documentos, consultas, qrels = datos['documentos'], datos['consultas'], datos['qrels']
    else:
        documentos, consultas, qrels = cargar_fiqa(ROOT / 'data')
        guardar(cache, {'documentos': documentos, 'consultas': consultas, 'qrels': qrels})
    if args.limite:
        documentos = dict(list(documentos.items())[:args.limite])
    fingerprint = huella(documentos)
    guardar(salida / 'corpus.json.gz', {
        'documentos': documentos, 'consultas': consultas, 'qrels': qrels,
        'huella': fingerprint, 'completo': not args.limite and not args.carpeta_txt,
        'origen': 'txt' if args.carpeta_txt else 'FiQA',
    })
    metodos = [args.metodo] if args.metodo else list(METODOS)
    print(f'Preparando {metodos}: {len(documentos)} documentos', flush=True)
    indices = construir_varios(procesar_documentos_varios(documentos, metodos, args.workers), metodos, fingerprint)
    for metodo, datos in indices.items():
        guardar(salida / f'indice_{metodo}.json.gz', datos)
        print(f'Vocabulario: {len(datos["indice"])}', flush=True)
    # Invalidar selección y evaluaciones anteriores si cambia el corpus.
    seleccion = salida / 'seleccion.json.gz'
    if seleccion.exists() and (leer(seleccion).get('huella') != fingerprint
                               or leer(seleccion).get('version_modelos') != 2):
        seleccion.unlink()
    print('Índices guardados. Siguiente paso: comparar_preprocesamiento.', flush=True)


if __name__ == '__main__':
    main()
