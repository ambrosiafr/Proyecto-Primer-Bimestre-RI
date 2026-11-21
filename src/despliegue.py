"""Obtiene un paquete público verificable al arrancar un alojamiento efímero."""
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def obtener_artefactos(carpeta):
    url = os.environ.get('ARTIFACTS_URL', '')
    checksum = os.environ.get('ARTIFACTS_SHA256', '')
    carpeta = Path(carpeta)
    if (carpeta / 'seleccion.json.gz').exists() or not url:
        return
    if not url.startswith('https://') or len(checksum) != 64:
        raise ValueError('Configura ARTIFACTS_URL HTTPS y ARTIFACTS_SHA256')
    with tempfile.TemporaryDirectory() as tmp:
        paquete = Path(tmp) / 'artefactos.zip'
        urllib.request.urlretrieve(url, paquete)
        with paquete.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != checksum.lower():
                raise ValueError('SHA256 de artefactos incorrecto')
        with zipfile.ZipFile(paquete) as z:
            for nombre in z.namelist():
                if Path(nombre).name != nombre or '..' in nombre or ':' in nombre or '\\' in nombre:
                    raise ValueError('Ruta inválida dentro del paquete')
            z.extractall(tmp + '/extraido')
        carpeta.mkdir(parents=True, exist_ok=True)
        # Selección se publica al final para señalar que el paquete está completo.
        archivos = list((Path(tmp) / 'extraido').iterdir())
        for archivo in sorted(archivos, key=lambda p: p.name == 'seleccion.json.gz'):
            archivo.replace(carpeta / archivo.name)
