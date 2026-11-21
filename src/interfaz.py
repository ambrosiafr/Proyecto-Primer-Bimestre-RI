"""Pantalla única de búsqueda, comparación y evaluación."""
import html
import time
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from src.analisis import por_tipo
from src.archivos import ROOT, leer
from src.presentacion import descripcion_breve, titulo_resultado


def ejecutar_consulta(motor, texto, qid, modelos, k):
    resultados, errores, tiempos = {}, {}, {}
    salida = {'texto': texto, 'qid': qid, 'k': k, 'rankings': resultados,
              'errores': errores, 'tiempos': tiempos, 'comparacion': len(modelos) > 1,
              'principal': modelos[0]}
    with st.spinner('Buscando…'):
        for modelo in modelos:
            try:
                inicio = time.perf_counter()
                resultados[modelo] = motor.buscar(texto, modelo, k)
                tiempos[modelo] = (time.perf_counter() - inicio) * 1000
            except (OSError, ValueError, ImportError, RuntimeError) as error:
                errores[modelo] = str(error)
    return salida


def leer_evaluacion(motor):
    ruta = ROOT / 'artefactos/evaluacion_test.json.gz'
    if not ruta.exists():
        return None
    reporte = leer(ruta)
    if (reporte['huella'] != motor.datos['huella'] or reporte['metodo'] != motor.metodo
            or reporte.get('split') != 'test' or not reporte.get('completo')
            or reporte.get('version_modelos') != 2
            or set(reporte['modelos']) != {'jaccard', 'tfidf', 'bm25', 'embeddings'}):
        return None
    qids = set(motor.datos['qrels']['test'])
    for modelo in reporte['modelos'].values():
        consultas = modelo['consultas']
        if len(consultas) != len(qids) or {f['qid'] for f in consultas} != qids:
            return None
    return reporte


def cargar_caso(motor, qid, k):
    """Carga el texto y top-n de una consulta enlazada en el estado del formulario."""
    st.session_state['consulta_texto'] = motor.datos['consultas'][qid]
    st.session_state['top_n'] = k
    st.session_state['caso_activo'] = qid
    st.session_state['caso_pendiente'] = qid


def consumir_enlace(motor):
    if 'consulta' not in st.query_params:
        return
    qid = st.query_params['consulta']
    try:
        k = max(1, min(50, int(st.query_params.get('top', '10'))))
    except (TypeError, ValueError):
        k = 10
    if qid in motor.datos['qrels'].get('test', {}):
        cargar_caso(motor, qid, k)
    else:
        st.warning('La consulta enlazada no pertenece al conjunto de prueba.')
    # El enlace se consume una vez y admite aperturas posteriores del mismo caso.
    del st.query_params['consulta']
    if 'top' in st.query_params:
        del st.query_params['top']


def identificar_consulta(motor, texto):
    activo = st.session_state.get('caso_activo')
    if activo and texto.strip() == motor.datos['consultas'][activo].strip():
        return activo
    coincidencias = [qid for qid in motor.datos['qrels'].get('test', {})
                     if motor.datos['consultas'][qid].strip() == texto.strip()]
    return coincidencias[0] if len(coincidencias) == 1 else None


def matriz_rankings(salida, nombres):
    filas = []
    for indice in range(salida['k']):
        fila = {'Posición': indice + 1}
        for modelo, nombre in nombres.items():
            ranking = salida['rankings'].get(modelo, [])
            if indice < len(ranking):
                docid, puntaje = ranking[indice]
                fila[nombre] = f'ID: {docid} · Puntaje: {puntaje:.4f}'
            else:
                fila[nombre] = '—'
        filas.append(fila)
    return pd.DataFrame(filas)


def mostrar_consulta(motor, salida, nombres):
    if not salida:
        return
    mostrar_principal(motor, salida, nombres)
    mostrar_comparacion(salida, nombres)


def mostrar_principal(motor, salida, nombres):
    st.subheader(f'Resultados: {salida["texto"]}')
    principal = salida.get('principal', next(iter(salida['rankings']), None))
    if principal in salida['errores']:
        st.error(f'{nombres[principal]}: {salida["errores"][principal]}')
    if principal in salida['rankings']:
        ranking = salida['rankings'][principal]
        if not ranking:
            st.info('Sin coincidencias.')
        for posicion, (docid, score) in enumerate(ranking, 1):
            texto = motor.datos['documentos'][docid]
            with st.container(border=True):
                titulo = titulo_resultado(posicion, docid, score, texto)
                st.markdown(f'<strong>{html.escape(titulo)}</strong>', unsafe_allow_html=True)
                st.text(descripcion_breve(texto, 220))
                with st.expander('Ver texto completo', expanded=False):
                    st.text(texto)

def mostrar_comparacion(salida, nombres):
    for modelo, error in salida['errores'].items():
        if modelo != salida.get('principal'):
            st.error(f'{nombres[modelo]}: {error}')
    with st.expander('Detalles de la consulta realizada', expanded=False):
        columnas = {'Posición': st.column_config.NumberColumn(width='small')}
        columnas.update({nombre: st.column_config.TextColumn(width='medium') for nombre in nombres.values()})
        st.dataframe(matriz_rankings(salida, nombres), hide_index=True, width='stretch',
                     column_config=columnas, key='rankings_detalle')


def mostrar_evaluacion(motor, reporte, nombres):
    st.subheader('Evaluación del sistema')
    if not reporte:
        st.info('La evaluación completa todavía no está disponible.')
        return
    k = reporte['k']
    resumen = [{'Modelo': nombres[m], f'Precision@{k}': d['resumen'][f'Precision@{k}'],
                f'Recall@{k}': d['resumen'][f'Recall@{k}'], 'MAP': d['resumen']['MAP'],
                'Tiempo (ms)': d['resumen']['ms_por_consulta']}
               for m, d in reporte['modelos'].items()]
    st.dataframe(pd.DataFrame(resumen), hide_index=True, width='stretch', key='evaluacion_global')

    filas = [{'Partición': 'test', 'ID consulta': f['qid'], 'Modelo': nombres[m], 'Consulta': f['consulta'], f'Precision@{k}': f[f'Precision@{k}'],
              f'Recall@{k}': f[f'Recall@{k}'], 'AP': f['AP'], 'Tiempo (ms)': f['ms']}
             for m, d in reporte['modelos'].items() for f in d['consultas']]
    tabla = pd.DataFrame(filas)
    with st.expander('Detalle de métricas por consulta', expanded=False):
        mostrar_detalle_metricas(motor, reporte, nombres)
        st.caption('38 registros carecen de texto; uno tiene relevancia positiva en test y se conserva '
                   'en la evaluación. Los qrels incompletos y el truncamiento semántico a 256 tokens limitan '
                   'la interpretación de los resultados.')
    st.download_button('Descargar métricas CSV', tabla.to_csv(index=False).encode('utf-8'),
                       'metricas.csv', 'text/csv', key='descargar_metricas')

    st.subheader('Comparación por tipo de consulta')
    tipos = pd.DataFrame(por_tipo(reporte))
    matriz = tipos.pivot(index='tipo', columns='modelo', values='MAP').rename(columns=nombres)
    matriz = matriz.reindex(columns=list(nombres.values()))
    matriz.insert(0, 'Consultas', tipos.groupby('tipo')['consultas'].first())
    matriz.index.name = 'Tipo de consulta'
    st.dataframe(matriz, width='stretch', key='comparacion_tipos')

    st.subheader('Dónde cambia el resultado')
    bm = {f['qid']: f['AP'] for f in reporte['modelos']['bm25']['consultas']}
    diferencias = [dict(qid=f['qid'], consulta=f['consulta'], delta=f['AP'] - bm[f['qid']])
                   for f in reporte['modelos']['embeddings']['consultas']]
    ganadas = sum(f['delta'] > 0 for f in diferencias)
    perdidas = sum(f['delta'] < 0 for f in diferencias)
    st.caption(f'Embeddings frente a BM25: {ganadas} mejoras · {perdidas} pérdidas · '
               f'{len(diferencias)-ganadas-perdidas} empates. ΔAP completo = AP embeddings − AP BM25. '
               'Pulsa una consulta para comparar sus rankings.')
    peores = sorted((f for f in diferencias if f['delta'] < 0), key=lambda f: (f['delta'], f['qid']))[:5]
    mejores = sorted((f for f in diferencias if f['delta'] > 0), key=lambda f: (-f['delta'], f['qid']))[:5]
    for columna, titulo, casos in zip(st.columns(2), ['Cinco mayores pérdidas', 'Cinco mayores mejoras'], [peores, mejores]):
        with columna:
            st.markdown(f'**{titulo}**')
            for caso in casos:
                href = '?' + urlencode({'consulta': caso['qid'], 'top': k}) + '#busqueda'
                st.markdown(f'<a href="{html.escape(href, quote=True)}" target="_self">'
                            f'{html.escape(caso["consulta"])}</a> '
                            f'<span>· ΔAP {caso["delta"]:+.4f}</span>', unsafe_allow_html=True)


def mostrar_detalle_metricas(motor, reporte, nombres):
    qrels = motor.datos['qrels']['test']
    opciones = list(qrels)
    juicios = sum(len(j) for j in qrels.values())
    st.caption(f'{len(opciones)} consultas de test · {juicios:,} juicios de relevancia.'.replace(',', '.'))
    if st.session_state.get('metricas_consulta') not in opciones:
        st.session_state['metricas_consulta'] = opciones[0]
    qid = st.selectbox('Consulta de prueba', opciones,
                       format_func=lambda q: motor.datos['consultas'][q], key='metricas_consulta')
    k = reporte['k']
    # Son las mismas métricas precalculadas del informe final, para todos los qids de test.
    filas = [{'modelo': m, **next(f for f in reporte['modelos'][m]['consultas'] if f['qid'] == qid)}
             for m in nombres]
    resumen = [{'Modelo': nombres[f['modelo']], f'Precision@{k}': f[f'Precision@{k}'],
                f'Recall@{k}': f[f'Recall@{k}'], 'AP': f['AP']} for f in filas]
    st.dataframe(pd.DataFrame(resumen), hide_index=True, width='stretch', key='metricas_consultas')
    st.caption(f'ID de consulta: {qid} · '
               f'Documentos relevantes: {sum(v > 0 for v in qrels[qid].values())}. '
               f'Precision/Recall usan los primeros {k} resultados; AP usa el ranking completo.')
    with st.expander('Documentos juzgados de esta consulta', expanded=False):
        st.dataframe(pd.DataFrame([{'ID documento': docid, 'Relevancia': score}
                                  for docid, score in qrels[qid].items()]),
                     hide_index=True, width='stretch', key='qrels_consulta')


def mostrar_pantalla(motor, nombres, seleccion):
    reporte = leer_evaluacion(motor)
    consumir_enlace(motor)
    if 'top_n' not in st.session_state:
        st.session_state['top_n'] = 10
    st.markdown('<div id="busqueda"></div>', unsafe_allow_html=True)
    texto = st.text_input('Escribe tu consulta', placeholder='How do index funds work?',
                          max_chars=2000, key='consulta_texto')
    modelo = st.selectbox('Modelo de recuperación', list(nombres), index=2,
                          format_func=lambda m: nombres[m], key='modelo_busqueda')
    k = st.slider('Top-n de resultados', min_value=1, max_value=50, key='top_n')
    buscar = st.button('Buscar documentos', type='primary', key='buscar')

    pendiente = st.session_state.pop('caso_pendiente', None)
    renderizado = False
    if pendiente or buscar:
        if texto.strip():
            qid = pendiente or identificar_consulta(motor, texto)
            # Las tarjetas del modelo elegido aparecen antes de completar la comparación.
            salida = ejecutar_consulta(motor, texto, qid, [modelo], int(k))
            mostrar_principal(motor, salida, nombres)
            otros = [m for m in nombres if m != modelo]
            comparacion = ejecutar_consulta(motor, texto, qid, otros, int(k))
            for campo in ('rankings', 'errores', 'tiempos'):
                salida[campo].update(comparacion[campo])
            salida['comparacion'] = True
            mostrar_comparacion(salida, nombres)
            renderizado = True
            st.session_state['resultado_actual'] = salida
        else:
            st.warning('Escribe una consulta.')
    salida = st.session_state.get('resultado_actual')
    if salida:
        intentados = set(salida['rankings']) | set(salida['errores'])
        if intentados != set(nombres):
            # Compatibilidad con resultados guardados que contienen un único modelo.
            principal = salida.get('principal', modelo)
            orden = [principal, *[m for m in nombres if m != principal]]
            salida = ejecutar_consulta(motor, salida['texto'], salida['qid'], orden, salida['k'])
            st.session_state['resultado_actual'] = salida
    if not renderizado:
        mostrar_consulta(motor, salida, nombres)
    st.divider()
    mostrar_evaluacion(motor, reporte, nombres)
