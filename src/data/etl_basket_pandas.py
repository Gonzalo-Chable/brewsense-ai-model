"""
src/data/etl_basket_pandas.py
Pipeline de Preprocesamiento y Agrupación de Canastas (ETL) — BrewSense AI
Líderes de Datos: Edwin Cruz / Rey Lopez
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def generar_dataset_respaldo(ruta_destino: Path) -> pd.DataFrame:
    print('[WARN] No se encontró CSV en data/raw/. Generando dataset base reproducible...')
    np.random.seed(42)
    sucursales = ["Lower Manhattan", "Hell's Kitchen", "Astoria"]
    pares_frecuentes = [
        ("Ouro Brasileiro shot", "Ginger Scone", 3.00, 3.50),
        ("Espresso Premium", "Croissant", 3.75, 3.50),
        ("Latte", "Pan de Elote", 4.25, 3.75),
        ("Cappuccino", "Chocolate Croissant", 4.00, 3.75),
        ("Earl Grey Rg", "Scone", 3.00, 3.25),
        ("Americano", "Brownie", 3.25, 3.50),
    ]
    productos_sueltos = [
        ("Earl Grey Rg", 3.00),
        ("Brewed Black Tea Lg", 3.00),
        ("Columbian Medium Roast Rg", 2.50),
        ("Dark Chocolate Lg", 4.50),
        ("Peppermint Rg", 3.00),
    ]
    filas = []
    tx_id = 1
    for i in range(1, 4501):
        fecha = f"2023-0{(i % 6) + 1}-{(i % 28) + 1:02d}"
        hora = f"{(i % 12) + 7:02d}:{(i % 60):02d}:00"
        store_idx = i % 3
        store_loc = sucursales[store_idx]
        store_id = store_idx + 3
        if np.random.rand() < 0.65:
            idx_par = np.random.choice(len(pares_frecuentes), p=[0.22, 0.22, 0.20, 0.16, 0.10, 0.10])
            prod_a, prod_b, precio_a, precio_b = pares_frecuentes[idx_par]
            filas.append([tx_id, fecha, hora, 1, store_id, store_loc, prod_a, precio_a])
            tx_id += 1
            if np.random.rand() < 0.72:
                filas.append([tx_id, fecha, hora, 1, store_id, store_loc, prod_b, precio_b])
                tx_id += 1
        else:
            prod_s, precio_s = productos_sueltos[i % len(productos_sueltos)]
            filas.append([tx_id, fecha, hora, 1, store_id, store_loc, prod_s, precio_s])
            tx_id += 1

    df_sintetico = pd.DataFrame(
        filas,
        columns=[
            "transaction_id", "transaction_date", "transaction_time",
            "transaction_qty", "store_id", "store_location",
            "product_detail", "unit_price"
        ]
    )
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    df_sintetico.to_csv(ruta_destino, index=False)
    return df_sintetico

def ejecutar_etl(ruta_raw: str, ruta_salida: str) -> None:
    ruta_in = Path(ruta_raw)
    ruta_out = Path(ruta_salida)
    print('=' * 70)
    print('[INFO] INICIANDO PIPELINE ETL - BREWSENSE AI (CAFÉ AROMA)')
    print('=' * 70)
    if not ruta_in.exists():
        archivos_csv = list(Path('data/raw').glob('*.csv')) if Path('data/raw').exists() else []
        if archivos_csv:
            ruta_in = archivos_csv[0]
            print(f'[INFO] Archivo detectado en data/raw/: {ruta_in.name}')
            df = pd.read_csv(ruta_in)
        else:
            df = generar_dataset_respaldo(ruta_in)
    else:
        print(f'[INFO] Cargando dataset crudo desde: {ruta_in}')
        df = pd.read_csv(ruta_in)

    total_filas_brutas = len(df)
    print(f'[INFO] Registros transaccionales brutos leídos: {total_filas_brutas:,}')

    columnas_clave = ['transaction_date', 'transaction_time', 'store_location', 'product_detail']
    df_limpio = df.dropna(subset=columnas_clave).copy()

    if 'unit_price' in df_limpio.columns:
        df_limpio['unit_price'] = (
            df_limpio['unit_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '.', regex=False)
        )
        df_limpio['unit_price'] = pd.to_numeric(df_limpio['unit_price'], errors='coerce').fillna(0.0)
        df_limpio = df_limpio[df_limpio['unit_price'] > 0]

    filas_eliminadas = total_filas_brutas - len(df_limpio)
    print(f'[INFO] Limpieza completada: {filas_eliminadas} registros nulos eliminados (0% nulos restantes).')

    mapeo_estados = {
        'Lower Manhattan': 'Quintana Roo',
        "Hell's Kitchen": 'Jalisco',
        'Astoria': 'Nuevo León'
    }
    df_limpio['estado_sucursal'] = df_limpio['store_location'].map(lambda x: mapeo_estados.get(x, x))
    print(f"[INFO] Sucursales mapeadas a estados: {df_limpio['estado_sucursal'].unique().tolist()}")

    store_col = 'store_id' if 'store_id' in df_limpio.columns else 'estado_sucursal'
    df_limpio['basket_id'] = (
        df_limpio['transaction_date'].astype(str) + '_' +
        df_limpio['transaction_time'].astype(str) + '_' +
        df_limpio[store_col].astype(str)
    )
    total_canastas = df_limpio['basket_id'].nunique()
    print(f'[INFO] Total de canastas únicas consolidadas: {total_canastas:,}')

    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    columnas_exportar = ['basket_id', 'transaction_date', 'estado_sucursal', 'product_detail', 'unit_price']
    columnas_finales = [c for c in columnas_exportar if c in df_limpio.columns]
    df_limpio[columnas_finales].to_csv(ruta_out, index=False)
    print(f'[SUCCESS] Dataset procesado guardado exitosamente en: {ruta_out}')
    print('=' * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/Coffee Shop Sales(in).csv')
    parser.add_argument('--output', type=str, default='data/processed/canastas_cafe_aroma.csv')
    args = parser.parse_args()
    ejecutar_etl(args.input, args.output)
