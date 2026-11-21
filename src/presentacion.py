"""Encabezados extractivos: FiQA no incluye autores y sus títulos están vacíos."""
import re
import textwrap


def encabezado_documento(texto, ancho=110):
    limpio = ' '.join(texto.split()).strip('" ')
    if not limpio:
        return 'Registro sin texto'
    primera = re.split(r'(?<=[.!?])\s+', limpio, maxsplit=1)[0]
    return textwrap.shorten(primera, width=ancho, placeholder='…')


def descripcion_breve(texto, ancho=360):
    return textwrap.shorten(' '.join(texto.split()), width=ancho, placeholder='…')


def titulo_resultado(posicion, docid, score, texto):
    return f'{posicion}. {encabezado_documento(texto, 64)} · ID: {docid} · Puntaje: {score:.4f}'
