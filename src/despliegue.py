"""Descarga los datos preparados cuando no están disponibles localmente."""
import gzip
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def obtener_artefactos(carpeta):
    carpeta = Path(carpeta)
    # Selección se instala al final y señala que terminó la descarga.
    if (carpeta / 'seleccion.json.gz').exists():
        return
    url = os.environ.get('ARTIFACTS_URL', '')
    if not url:
        return
    if not url.startswith('https://'):
        raise ValueError('ARTIFACTS_URL debe ser un enlace HTTPS al ZIP')
    carpeta.parent.mkdir(parents=True, exist_ok=True)
    # El temporal comparte disco con el destino para permitir replace.
    with tempfile.TemporaryDirectory(dir=carpeta.parent) as temporal:
        temporal = Path(temporal)
        paquete = temporal / 'artefactos.zip'
        urllib.request.urlretrieve(url, paquete)
        with zipfile.ZipFile(paquete) as zip_datos:
            for nombre in zip_datos.namelist():
                if Path(nombre).name != nombre or '..' in nombre or ':' in nombre or '\\' in nombre:
                    raise ValueError('Ruta inválida dentro del paquete')
            zip_datos.extractall(temporal / 'extraido')
        extraido = temporal / 'extraido'
        with gzip.open(extraido / 'seleccion.json.gz', 'rt', encoding='utf-8') as archivo:
            seleccion = json.load(archivo)
        if seleccion['metodo'] not in ('stemming', 'lematizacion'):
            raise ValueError('Método de preprocesamiento inválido en el paquete')
        necesarios = ['corpus.json.gz', 'seleccion.json.gz',
                      f'indice_{seleccion["metodo"]}.json.gz',
                      'vectores.faiss', 'vectores.json', 'evaluacion_test.json.gz']
        if not all((extraido / nombre).is_file() for nombre in necesarios):
            raise ValueError('El ZIP no contiene todos los datos preparados')
        carpeta.mkdir(parents=True, exist_ok=True)
        archivos = sorted(extraido.iterdir(), key=lambda p: p.name == 'seleccion.json.gz')
        for archivo in archivos:
            archivo.replace(carpeta / archivo.name)
