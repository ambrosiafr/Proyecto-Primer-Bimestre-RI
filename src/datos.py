"""Descarga los Parquet oficiales y qrels TSV; conserva los IDs originales."""
import csv
import json
from pathlib import Path


def cargar_fiqa(destino):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo = 'BeIR/fiqa'
    revision = api.dataset_info(repo).sha
    archivos = api.list_repo_files(repo, repo_type='dataset', revision=revision)
    resultados = []
    for configuracion in ('corpus', 'queries'):
        filas = {}
        for nombre in sorted(archivos):
            if nombre.startswith(configuracion + '/') and nombre.endswith('.parquet'):
                ruta = hf_hub_download(repo, nombre, repo_type='dataset', revision=revision)
                for fila in pd.read_parquet(ruta).to_dict('records'):
                    docid = str(fila['_id'])
                    texto = fila.get('text')
                    if not texto:
                        texto = ''
                    texto = str(texto)
                    if configuracion == 'corpus':
                        titulo = fila.get('title')
                        if not titulo:
                            titulo = ''
                        texto = (str(titulo) + ' ' + texto).strip()
                    if docid in filas:
                        raise ValueError(f'ID duplicado: {docid}')
                    filas[docid] = texto
        if not filas:
            raise ValueError(f'No se encontró {configuracion} en {repo}')
        resultados.append(filas)
    qrevision = api.dataset_info('BeIR/fiqa-qrels').sha
    qrels = {}
    for split in ('train', 'dev', 'test'):
        ruta = hf_hub_download('BeIR/fiqa-qrels', f'{split}.tsv', repo_type='dataset', revision=qrevision)
        qrels[split] = leer_qrels(ruta)
    documentos, consultas = resultados
    for split, juicios in qrels.items():
        for qid, relevantes in juicios.items():
            if qid not in consultas or not set(relevantes).issubset(documentos):
                raise ValueError(f'IDs de {split} sin correspondencia con FiQA')
    (destino / 'fuentes.json').write_text(json.dumps({
        'corpus_repo': repo, 'revision': revision,
        'qrels_repo': 'BeIR/fiqa-qrels', 'qrels_revision': qrevision,
        'licencia_dataset': 'CC-BY-SA-4.0'
    }, indent=2), encoding='utf-8')
    return documentos, consultas, qrels


def leer_qrels(ruta):
    resultado = {}
    with open(ruta, encoding='utf-8') as f:
        for fila in csv.DictReader(f, delimiter='\t'):
            qid = str(fila['query-id'])
            docid = str(fila['corpus-id'])
            relevancia = int(fila['score'])
            if qid not in resultado:
                resultado[qid] = {}
            resultado[qid][docid] = relevancia
    return resultado
