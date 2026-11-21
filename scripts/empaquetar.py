"""Paquete de artefactos para adjuntar a una GitHub Release (no al historial Git)."""
import hashlib
import zipfile
from src.archivos import ROOT, leer


def main():
    carpeta = ROOT / 'artefactos'
    seleccion = leer(carpeta / 'seleccion.json.gz')
    nombres = ['corpus.json.gz', 'seleccion.json.gz', f'indice_{seleccion["metodo"]}.json.gz',
               'vectores.faiss', 'vectores.json', 'evaluacion_test.json.gz']
    destino = ROOT / 'dist'
    destino.mkdir(exist_ok=True)
    ruta = destino / 'fiqa-artefactos.zip'
    with zipfile.ZipFile(ruta, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for nombre in nombres:
            z.write(carpeta / nombre, nombre)
        fuente = ROOT / 'data/fuentes.json'
        if fuente.exists():
            z.write(fuente, 'fuentes.json')
    with ruta.open('rb') as archivo:
        checksum = hashlib.file_digest(archivo, 'sha256').hexdigest()
    (destino / 'fiqa-artefactos.sha256').write_text(checksum, encoding='ascii')
    print(ruta, checksum)
    fuentes = ['app.py', 'README.md', 'requirements.txt', 'requirements-lock.txt',
               'render.yaml', '.gitignore', 'src', 'scripts', '.github', '.streamlit/config.toml']
    internos = {'scripts/verificar_interfaz.py', '.github/workflows/pruebas.yml'}
    with zipfile.ZipFile(destino / 'fiqa-codigo.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for nombre in fuentes:
            origen = ROOT / nombre
            rutas = origen.rglob('*') if origen.is_dir() else [origen]
            for archivo in rutas:
                relativo = archivo.relative_to(ROOT).as_posix()
                if (archivo.is_file() and '__pycache__' not in archivo.parts
                        and archivo.suffix != '.pyc' and relativo not in internos):
                    z.write(archivo, archivo.relative_to(ROOT).as_posix())
    print('Código empaquetado:', destino / 'fiqa-codigo.zip')


if __name__ == '__main__':
    main()
