"""Registra ejemplos lingüísticos reales del preprocesador para la documentación."""
from src.archivos import ROOT
from src.preprocesamiento import METODOS, procesar_texto


def main():
    ejemplos = ["Banks are investing in investments.", "This income is not taxable.",
                "This income isn't taxable.", "My 401(k) invests in the S&P 500.",
                "A $1,500.50 loss is -10% of my savings."]
    lineas = ['# Ejemplos comprobados del preprocesamiento', '',
              'Textos de control inventados para inspección, no resultados de recuperación de FiQA.', '']
    for texto in ejemplos:
        lineas.extend(['## ' + texto, ''])
        for metodo in METODOS:
            lineas.append(f'- {metodo}: `{procesar_texto(texto, metodo)}`')
        lineas.append('')
    lineas.append('El tokenizador puede dividir expresiones financieras. Las mismas reglas se aplican '
                  'a documentos y consultas clásicas; embeddings utiliza texto original.')
    (ROOT / 'output').mkdir(exist_ok=True)
    (ROOT / 'output/ejemplos_preprocesamiento.md').write_text('\n'.join(lineas), encoding='utf-8')


if __name__ == '__main__':
    main()
