"""Evaluación exhaustiva: MAP de rankings completos, no MAP@10 renombrado."""
import argparse
import time
from src.archivos import ROOT, guardar
from src.evaluacion import metricas, resumir
from src.motor import Motor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--modelos', nargs='+', choices=['jaccard', 'tfidf', 'bm25', 'embeddings'],
                   default=['jaccard', 'tfidf', 'bm25', 'embeddings'])
    args = p.parse_args()
    motor = Motor(ROOT / 'artefactos')
    if not motor.datos['completo']:
        raise ValueError('Evaluar oficialmente exige el corpus FiQA completo')
    salida = {'split': args.split, 'k': args.k, 'completo': True,
              'bm25': {'k1': 1.2, 'b': 0.75, 'qf': False},
              'tfidf': 'tf crudo * (ln((N+1)/(DF+1))+1), coseno',
              'politica_vacios': 'vector_cero',
              'metodo': motor.metodo,
              'seleccion': motor.seleccion, 'modelos': {}}
    for modelo in args.modelos:
        # Excluir carga inicial de modelos del tiempo de búsqueda reportado.
        motor.buscar('bank account', modelo, 1)
        filas = []
        for qid, relevantes in motor.datos['qrels'][args.split].items():
            inicio = time.perf_counter()
            ranking = motor.buscar(motor.datos['consultas'][qid], modelo, completo=True)
            ms = (time.perf_counter() - inicio) * 1000
            if len(ranking) != len(motor.datos['documentos']):
                raise ValueError('MAP exige ranking completo')
            fila = {'qid': qid, 'consulta': motor.datos['consultas'][qid], 'ms': ms,
                    **metricas(ranking, relevantes, args.k, completo=True),
                    'top': ranking[:args.k]}
            filas.append(fila)
        salida['modelos'][modelo] = {'resumen': resumir(filas, args.k), 'consultas': filas}
        print(modelo, salida['modelos'][modelo]['resumen'], flush=True)
        guardar(ROOT / f'artefactos/evaluacion_{args.split}.json.gz', salida)


if __name__ == '__main__':
    main()
