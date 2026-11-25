"""Pipeline reproducible. No se ejecuta durante búsquedas de usuarios."""
import subprocess
import sys


def main():
    for modulo in ('preparar_datos', 'comparar_preprocesamiento', 'preparar_embeddings', 'evaluar'):
        subprocess.run([sys.executable, '-m', 'scripts.' + modulo], check=True)


if __name__ == '__main__':
    main()
