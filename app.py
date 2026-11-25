"""Punto de entrada de la interfaz web Streamlit."""
import os
import streamlit as st
from src.archivos import ROOT
from src.motor import Motor
from src.despliegue import obtener_artefactos
from src.interfaz import mostrar_pantalla

st.set_page_config(page_title='FiQA Sistema de Recuperación de Información', layout='wide')
st.title('FiQA Sistema de Recuperación de Información')
st.caption('Sebastián Chicaiza y Juan Murillo · Recuperación de Información · Primer bimestre')
CARPETA = ROOT / 'artefactos'
if 'ARTIFACTS_URL' not in os.environ:
    try:
        if 'ARTIFACTS_URL' in st.secrets:
            os.environ['ARTIFACTS_URL'] = str(st.secrets['ARTIFACTS_URL'])
    except FileNotFoundError:
        pass
try:
    with st.spinner('Preparando colección…'):
        obtener_artefactos(CARPETA)
except (OSError, ValueError) as error:
    st.error(f'No se pudo descargar la colección: {error}')
    st.stop()
NOMBRES = {'jaccard': 'Binario · Jaccard', 'tfidf': 'Vectorial · TF-IDF',
           'bm25': 'Probabilístico · BM25', 'embeddings': 'Semántico · Embeddings'}


@st.cache_resource
def cargar_motor():
    return Motor(CARPETA)


if not (CARPETA / 'seleccion.json.gz').exists():
    st.info('Falta preparar los datos. Ejecuta el pipeline desde la terminal para habilitar el buscador.')
    st.code('python -m scripts.preparar_todo', language='bash')
    st.markdown('Las instrucciones completas están en el README del repositorio.')
    st.stop()

try:
    motor = cargar_motor()
except (OSError, ValueError) as error:
    st.error(f'No se pudieron cargar los artefactos: {error}')
    st.stop()

mostrar_pantalla(motor, NOMBRES)
