"""JSON comprimido: sin deserialización ejecutable mediante pickle."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def guardar(ruta, objeto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    with gzip.open(temporal, 'wt', encoding='utf-8') as f:
        json.dump(objeto, f, ensure_ascii=False)
    temporal.replace(ruta)


def leer(ruta):
    with gzip.open(ruta, 'rt', encoding='utf-8') as f:
        return json.load(f)
