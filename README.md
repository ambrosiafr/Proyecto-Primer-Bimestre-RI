# FiQA Sistema de Recuperación de Información

**Integrantes:** Sebastián Chicaiza y Juan Murillo.

**Página del Proyecto:** https://proyecto-primer-bimestre-ri-jywrddu9m56mmfodc4qjea.streamlit.app/

Aplicación en Python y Streamlit para comparar Jaccard binario, TF-IDF con coseno, BM25 y embeddings con FAISS. Los modelos clásicos utilizan un índice invertido y fórmulas implementadas con bucles y diccionarios. La búsqueda utiliza los **57.638 documentos del corpus FiQA**.

## Requisitos

- Python 3.11.
- Espacio para dependencias, corpus e índices. El procesamiento utiliza CPU.

## Instalación

Desde la carpeta del proyecto, en PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux, el entorno se crea con `python3.11 -m venv .venv`; el ejecutable queda en `.venv/bin/python`.

El modelo de spaCy se instala junto con las dependencias. `requirements-lock.txt` registra el entorno probado en Windows.

## Preparación de los datos

La preparación completa descarga FiQA, construye los índices, compara stemming y lematización con dev, genera los embeddings y evalúa test:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_todo
```

Este proceso se realiza una vez; el tiempo depende del equipo y de las descargas. Las búsquedas posteriores reutilizan los datos y los índices guardados localmente.

Para indexar otro corpus de archivos `.txt` UTF-8:

```powershell
.\.venv\Scripts\python.exe -m scripts.preparar_datos --carpeta-txt ruta\documentos
```

Esta opción construye los índices del corpus propio; la evaluación incluida corresponde a FiQA y requiere sus consultas y qrels.

## Inicio de la aplicación

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La página se abre en `http://localhost:8501`. La terminal se mantiene abierta mientras se utiliza la aplicación. Ctrl+C detiene el servidor.

El formulario permite escribir una consulta en inglés, elegir un modelo y definir el top-n. Las tarjetas muestran los resultados del modelo elegido; el detalle compara los cuatro rankings. El apartado de evaluación incluye todas las consultas test y una descarga general de métricas.

La primera búsqueda carga spaCy y el codificador semántico. Las siguientes reutilizan esos recursos. Después de actualizar el código, se reinicia Streamlit para descartar los recursos anteriores.

## Modelos y evaluación

- **Jaccard:** intersección dividida por unión de términos únicos.
- **TF-IDF:** frecuencia cruda e IDF suavizada `ln((N+1)/(df+1))+1`, con similitud de coseno.
- **BM25:** k1=1.2 y b=0.75, con términos únicos de consulta.
- **Embeddings:** all-MiniLM-L6-v2, 384 dimensiones y máximo 256 tokens. FAISS IndexFlatIP calcula coseno mediante vectores normalizados.

Stemming obtuvo BM25 MAP@10 de 0.208685 en dev, frente a 0.196503 con lematización. Por ese resultado se utiliza stemming en los tres modelos clásicos. Embeddings recibe el texto original.

La evaluación final utiliza **648 consultas test y 1.706 qrels** sobre el corpus completo. Train no se utiliza para entrenar los modelos y dev se reserva a la selección del preprocesamiento.

| Modelo | Precision@10 media | Recall@10 medio | MAP completo |
|---|---:|---:|---:|
| Jaccard | 0.0159 | 0.0649 | 0.0339 |
| TF-IDF | 0.0511 | 0.2187 | 0.1354 |
| BM25 | 0.0677 | 0.3071 | 0.2026 |
| Embeddings | 0.1045 | 0.4413 | 0.3117 |

Precision@10 divide los aciertos entre diez; Recall@10 los divide entre todos los relevantes. AP promedia la precisión en las posiciones relevantes del ranking completo y MAP promedia AP entre consultas. Los empates se ordenan por ID textual ascendente. Para evaluar el ranking completo, los modelos clásicos incluyen también los documentos con puntaje cero.

Los documentos sin juicio se tratan como no relevantes para las métricas. Se conservan los 38 registros vacíos del corpus; sus vectores son cero y no aparecen como resultados interactivos. Los qrels sobre ellos permanecen en la evaluación. Los títulos originales están vacíos, por lo que las tarjetas utilizan fragmentos del texto como encabezados.

## Repositorio

El repositorio contiene el código de la aplicación, su configuración y este README. El informe técnico se entrega por separado. Los datos, los índices, el entorno virtual y los archivos internos quedan fuera del historial de Git. Los datos necesarios para ejecutar la aplicación se generan con el comando de preparación indicado arriba.

## Estructura

```text
app.py           Inicio de la aplicación
src/             Datos, procesamiento, índices, modelos, métricas e interfaz
scripts/         Preparación de datos y evaluación
.streamlit/      Configuración visual
.github/         Preparación del corpus en GitHub Actions
render.yaml      Configuración de Render
```

## Fuentes y licencias

- [Corpus FiQA](https://huggingface.co/datasets/BeIR/fiqa) y [qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), bajo CC-BY-SA-4.0.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), bajo Apache-2.0.

Los datos y modelos se utilizan conforme a las condiciones de sus licencias.
