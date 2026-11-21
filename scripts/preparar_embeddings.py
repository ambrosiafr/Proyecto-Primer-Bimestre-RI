import argparse
from src.archivos import ROOT, leer
from src.modelos.embeddings import preparar


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size debe ser positivo')
    corpus = leer(ROOT / 'artefactos/corpus.json.gz')
    preparar(corpus['documentos'], ROOT / 'artefactos', args.batch_size)


if __name__ == '__main__':
    main()
