# BrewSense AI — Motor de Inteligencia de Ventas (Entrega 3)

Motor de minería de reglas de asociación (*Apriori* vs. *FP-Growth*) para la detección de productos ancla y recomendación de combos en **Café Aroma**.

## Integrantes y Roles
* **Edwin Cruz / Rey Lopez:** Líderes de Datos y Preprocesamiento
* **Rusel Tec:** Líder de Modelado y Experimentación
* **Gonzalo Chable / Uzias Perez:** Líderes de Pipeline e Integración

## Comando de Reproducibilidad en una Sola Línea
```bash
python src/models/train_association_rules.py --data data/processed/canastas_cafe_aroma.csv --min_support 0.05 --min_confidence 0.60 --output models/
```
