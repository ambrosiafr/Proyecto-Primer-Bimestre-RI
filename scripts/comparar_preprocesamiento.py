"""Selecciona SOLO con dev y MAP@10 de BM25. Desempate: stemming."""
import argparse
from src.archivos import ROOT, guardar, leer
from src.evaluacion import metricas
from src.modelos.clasicos import ModelosClasicos
from src.preprocesamiento import METODOS, procesar_texto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=10)
    args = p.parse_args()
    carpeta = ROOT / 'artefactos'
    corpus = leer(carpeta / 'corpus.json.gz')
    if not corpus['completo']:
        raise ValueError('La selección exige el corpus FiQA completo')
    resultados = {}
    for metodo in METODOS:
        datos = leer(carpeta / f'indice_{metodo}.json.gz')
        motor = ModelosClasicos(datos)
        aps = []
        for qid, relevantes in corpus['qrels']['dev'].items():
            tokens = procesar_texto(corpus['consultas'][qid], metodo)
            ranking = motor.buscar(tokens, 'bm25', args.k)
            aps.append(metricas(ranking, relevantes, args.k)[f'AP@{args.k}'])
        resultados[metodo] = sum(aps) / len(aps)
        print(metodo, resultados[metodo], flush=True)
    elegido = max(METODOS, key=lambda m: resultados[m])
    guardar(carpeta / 'seleccion.json.gz', {
        'metodo': elegido, 'criterio': f'BM25 MAP@{args.k} en dev',
        'resultados': resultados,
        'consultas': len(corpus['qrels']['dev'])
    })
    print('Método seleccionado:', elegido)

if __name__ == '__main__':
    main()
