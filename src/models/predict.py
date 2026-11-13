"""
src/models/predict.py
Función de Inferencia Desacoplada (Contrato de Interfaz) — BrewSense AI
Líderes de Pipeline e Integración: Gonzalo Chable / Uzias Perez
"""
import json
from pathlib import Path
import joblib
import pandas as pd

RUTA_MODELO = Path(__file__).resolve().parents[2] / 'models' / 'reglas_fpgrowth_v1.joblib'
REGLAS_IA = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else pd.DataFrame()

def predecir(entrada: dict) -> dict:
    if REGLAS_IA.empty:
        return {'producto_base': entrada.get('producto_ancla', 'N/A'), 'etiqueta': 'Error: Modelo no cargado', 'recomendaciones': [], 'status': 'error'}

    producto_actual = str(entrada.get('producto_ancla', '')).strip()
    confianza_minima = float(entrada.get('min_confidence', 0.60))

    recomendaciones_df = REGLAS_IA[
        REGLAS_IA['antecedents'].apply(lambda items: any(producto_actual.lower() == str(item).lower() for item in list(items)))
    ].copy()
    recomendaciones_df = recomendaciones_df[recomendaciones_df['confidence'] >= confianza_minima]

    if recomendaciones_df.empty:
        return {
            'producto_base': producto_actual,
            'etiqueta': 'Venta Individual / Sin asociaciones fuertes',
            'recomendaciones': [],
            'status': 'warning_no_rules'
        }

    recomendaciones_df = recomendaciones_df.sort_values(by='lift', ascending=False)
    mejor_regla = recomendaciones_df.iloc[0]
    complemento = list(mejor_regla['consequents'])[0]

    return {
        'producto_base': producto_actual,
        'etiqueta': 'Producto Gancho Detectado',
        'recomendaciones': [
            {
                'producto_sugerido': complemento,
                'soporte_pct': round(float(mejor_regla['support']) * 100, 2),
                'confianza_pct': round(float(mejor_regla['confidence']) * 100, 2),
                'lift': round(float(mejor_regla['lift']), 2)
            }
        ],
        'status': 'success'
    }

if __name__ == '__main__':
    print('=' * 65)
    print('PRUEBA UNITARIA DE INFERENCIA DESACOPLADA (SECCIÓN 5.B)')
    print('=' * 65)
    ejemplo_entrada = {'producto_ancla': 'Ouro Brasileiro shot', 'min_confidence': 0.60}
    print('\n• Ejemplo de Entrada (JSON / Diccionario):')
    print(json.dumps(ejemplo_entrada, indent=2, ensure_ascii=False))
    resultado_real = predecir(ejemplo_entrada)
    print('\n• Salida Real Retornada por el Modelo FP-Growth:')
    print(json.dumps(resultado_real, indent=2, ensure_ascii=False))
    print('=' * 65)
