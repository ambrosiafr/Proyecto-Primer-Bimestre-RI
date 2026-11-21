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
        raise ValueError('La selección exige FiQA completo; el límite es solo para desarrollo')
    resultados = {}
    for metodo in METODOS:
        datos = leer(carpeta / f'indice_{metodo}.json.gz')
        if datos.get('version') != 2:
            raise ValueError('Ejecuta preparar_datos para actualizar las fórmulas')
        if datos['huella'] != corpus['huella']:
            raise ValueError('Índice de un corpus distinto')
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
        'version_modelos': 2,
        'resultados': resultados, 'huella': corpus['huella'],
        'consultas': len(corpus['qrels']['dev'])
    })
    print('Método seleccionado:', elegido)
    diferencia = abs(resultados['stemming'] - resultados['lematizacion'])
    razon = (f'Obtuvo el mayor BM25 MAP@{args.k} de validación, con diferencia absoluta {diferencia:.6f}.'
             if diferencia else 'Ambos empataron; se aplicó el desempate declarado a favor de stemming.')
    (ROOT / 'output').mkdir(exist_ok=True)
    (ROOT / 'output/resultados_validacion.md').write_text(
        '# Selección de preprocesamiento: resultado real\n\n'
        f'Corpus completo: {len(corpus["documentos"])} documentos. '
        f'Consultas dev: {len(corpus["qrels"]["dev"])}.\n\n'
        f'BM25 fijo: k1=1.2, b=0.75; términos únicos de consulta. '
        f'Criterio: MAP@{args.k}, AP@k dividido por el total de relevantes.\n\n'
        f'- Stemming: {resultados["stemming"]:.6f}\n'
        f'- Lematización: {resultados["lematizacion"]:.6f}\n\n'
        f'**Elegido: {elegido}.** {razon}\n\n'
        'Esta elección es empírica para FiQA y este protocolo; no implica que la técnica sea universalmente superior. '
        'La interpretabilidad de los lemas o la mayor reducción de las raíces son hipótesis lingüísticas, '
        'pero el motivo de elección final es el resultado en dev. Test no se utilizó para decidir.\n\n'
        f'Huella de corpus: {corpus["huella"]}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
