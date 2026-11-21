"""JSON comprimido: sin deserialización ejecutable mediante pickle."""
import gzip
import hashlib
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


def huella(documentos):
    h = hashlib.sha256()
    for docid, texto in documentos.items():
        h.update(json.dumps([docid, texto], ensure_ascii=False).encode('utf-8'))
    return h.hexdigest()
