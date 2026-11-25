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
    if (reporte.get('split') != 'test'
            or not reporte.get('completo')
            or reporte.get('metodo') != motor.metodo
            or set(reporte['modelos']) != {'jaccard', 'tfidf', 'bm25', 'embeddings'}):
        return None
    qids = set(motor.datos['qrels']['test'])
    for modelo in reporte['modelos'].values():
        consultas = modelo['consultas']
        ids_evaluados = set()
        for consulta in consultas:
            ids_evaluados.add(consulta['qid'])
        if len(consultas) != len(qids) or ids_evaluados != qids:
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
    coincidencias = []
    for qid in motor.datos['qrels'].get('test', {}):
        if motor.datos['consultas'][qid].strip() == texto.strip():
            coincidencias.append(qid)
    if len(coincidencias) == 1:
        return coincidencias[0]
    return None


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
    st.subheader('Resultados de búsqueda')
    st.caption(f'Consulta: “{salida["texto"]}”')
    principal = salida['principal']
    if principal in salida['errores']:
        st.error(f'{nombres[principal]}: {salida["errores"][principal]}')
    if principal in salida['rankings']:
        ranking = salida['rankings'][principal]
        duracion = salida.get('tiempos', {}).get(principal)
        if duracion is not None:
            st.caption(f'{len(ranking)} resultados · {nombres[principal]} · {duracion:.1f} ms')
        if not ranking:
            st.info('No se encontraron resultados para esta consulta. Prueba con otras palabras o elige otro modelo.')
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
        for nombre in nombres.values():
            columnas[nombre] = st.column_config.TextColumn(width='medium')
        st.dataframe(matriz_rankings(salida, nombres), hide_index=True, width='stretch',
                     column_config=columnas, key='rankings_detalle')


def mostrar_evaluacion(motor, reporte, nombres):
    st.subheader('Evaluación del sistema')
    if not reporte:
        st.info('La evaluación completa todavía no está disponible. Cuando se preparen los datos de prueba, aquí podrás comparar los cuatro modelos.')
        return
    k = reporte['k']
    resumen = []
    filas = []
    for modelo, datos in reporte['modelos'].items():
        metricas_modelo = datos['resumen']
        resumen.append({
            'Modelo': nombres[modelo],
            f'Precision@{k}': metricas_modelo[f'Precision@{k}'],
            f'Recall@{k}': metricas_modelo[f'Recall@{k}'],
            'MAP': metricas_modelo['MAP'],
            'Tiempo (ms)': metricas_modelo['ms_por_consulta']
        })
        for consulta in datos['consultas']:
            filas.append({
                'Partición': 'test',
                'ID consulta': consulta['qid'],
                'Modelo': nombres[modelo],
                'Consulta': consulta['consulta'],
                f'Precision@{k}': consulta[f'Precision@{k}'],
                f'Recall@{k}': consulta[f'Recall@{k}'],
                'AP': consulta['AP'],
                'Tiempo (ms)': consulta['ms']
            })
    st.dataframe(pd.DataFrame(resumen), hide_index=True, width='stretch', key='evaluacion_global')

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
    ap_bm25 = {}
    for consulta in reporte['modelos']['bm25']['consultas']:
        ap_bm25[consulta['qid']] = consulta['AP']

    mejoras = []
    perdidas = []
    empates = 0
    for consulta in reporte['modelos']['embeddings']['consultas']:
        qid = consulta['qid']
        diferencia = consulta['AP'] - ap_bm25[qid]
        caso = {'qid': qid, 'consulta': consulta['consulta'], 'delta': diferencia}
        if diferencia > 0:
            mejoras.append(caso)
        elif diferencia < 0:
            perdidas.append(caso)
        else:
            empates += 1
    ganadas = len(mejoras)
    cantidad_perdidas = len(perdidas)
    st.caption(f'Embeddings frente a BM25: {ganadas} mejoras · {cantidad_perdidas} pérdidas · '
               f'{empates} empates. ΔAP completo = AP embeddings − AP BM25. '
               'Pulsa una consulta para comparar sus rankings.')
    peores = sorted(perdidas, key=lambda caso: (caso['delta'], caso['qid']))[:5]
    mejores = sorted(mejoras, key=lambda caso: (-caso['delta'], caso['qid']))[:5]
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
    juicios = 0
    for documentos_relevantes in qrels.values():
        juicios += len(documentos_relevantes)
    st.caption(f'{len(opciones)} consultas de test · {juicios:,} juicios de relevancia.'.replace(',', '.'))
    if st.session_state.get('metricas_consulta') not in opciones:
        st.session_state['metricas_consulta'] = opciones[0]
    qid = st.selectbox('Consulta de prueba', opciones,
                       format_func=lambda q: motor.datos['consultas'][q], key='metricas_consulta')
    k = reporte['k']
    # Son las mismas métricas precalculadas del informe final, para todos los qids de test.
    resumen = []
    for modelo, nombre in nombres.items():
        for fila in reporte['modelos'][modelo]['consultas']:
            if fila['qid'] == qid:
                resumen.append({
                    'Modelo': nombre,
                    f'Precision@{k}': fila[f'Precision@{k}'],
                    f'Recall@{k}': fila[f'Recall@{k}'],
                    'AP': fila['AP']
                })
                break
    st.dataframe(pd.DataFrame(resumen), hide_index=True, width='stretch', key='metricas_consultas')
    cantidad_relevantes = 0
    documentos_juzgados = []
    for docid, relevancia in qrels[qid].items():
        if relevancia > 0:
            cantidad_relevantes += 1
        documentos_juzgados.append({'ID documento': docid, 'Relevancia': relevancia})
    st.caption(f'ID de consulta: {qid} · '
               f'Documentos relevantes: {cantidad_relevantes}. '
               f'Precision/Recall usan los primeros {k} resultados; AP usa el ranking completo.')
    with st.expander('Documentos juzgados de esta consulta', expanded=False):
        st.dataframe(pd.DataFrame(documentos_juzgados),
                     hide_index=True, width='stretch', key='qrels_consulta')


def mostrar_pantalla(motor, nombres):
    reporte = leer_evaluacion(motor)
    consumir_enlace(motor)
    if 'top_n' not in st.session_state:
        st.session_state['top_n'] = 10
    pesta_buscar, pesta_evaluar = st.tabs(['Buscar documentos', 'Evaluar modelos'])
    with pesta_buscar:
        st.markdown('<div id="busqueda"></div>', unsafe_allow_html=True)
        st.write('Escribe una pregunta en inglés y elige cómo ordenar los documentos.')
        texto = st.text_input('Tu consulta', placeholder='How do index funds work?',
                              help='Usa palabras o una pregunta en inglés. Máximo 2.000 caracteres.',
                              max_chars=2000, key='consulta_texto')
        modelo = st.selectbox('Modelo para ordenar los resultados', list(nombres), index=2,
                              format_func=lambda m: nombres[m], key='modelo_busqueda',
                              help='Jaccard y TF-IDF buscan coincidencias de términos; BM25 pondera su importancia; embeddings busca similitud de significado.')
        k = st.slider('Cantidad de resultados', min_value=1, max_value=50, key='top_n',
                      help='Número máximo de documentos que se mostrarán.')
        buscar = st.button('Buscar', type='primary', key='buscar', disabled=not bool(texto.strip()))
        if not texto.strip():
            st.caption('Escribe una consulta para habilitar la búsqueda.')

    with pesta_buscar:
        pendiente = st.session_state.pop('caso_pendiente', None)
        renderizado = False
        if pendiente or buscar:
            if texto.strip():
                qid = pendiente or identificar_consulta(motor, texto)
                salida = ejecutar_consulta(motor, texto, qid, [modelo], int(k))
                mostrar_principal(motor, salida, nombres)
                otros = []
                for otro_modelo in nombres:
                    if otro_modelo != modelo:
                        otros.append(otro_modelo)
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
        if salida and not renderizado:
            coincide = (texto.strip() == salida['texto'].strip()
                        and modelo == salida.get('principal')
                        and int(k) == salida.get('k'))
            if not coincide:
                st.info('Cambiaste la consulta o sus opciones. Pulsa Buscar para actualizar los resultados.')
                salida = None
        if not renderizado:
            mostrar_consulta(motor, salida, nombres)
    with pesta_evaluar:
        mostrar_evaluacion(motor, reporte, nombres)
