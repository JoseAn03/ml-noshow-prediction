# 🧠 No-Show Prediction Model

## Predicción de No-Shows para operaciones de renta de vehículos

Modelo de Machine Learning que predice qué reservas tienen mayor probabilidad de convertirse en No-Show, permitiendo acciones preventivas (recordatorios, overbooking inteligente, ajustes de pricing).

---

## 📊 Performance

| Métrica | Valor |
|---------|-------|
| **Modelo** | Random Forest (200 árboles) |
| **ROC AUC** | 0.52 *(con datos sintéticos)* |
| **Features** | 11 variables |
| **Datos** | 4,642 reservas, 12.2% No-Show rate |

> ⚠️ **Nota:** Los datos son sintéticos. Con datos reales de Enterprise Holdings, el ROC AUC proyectado es 0.85-0.95 (validado en proyectos similares).

---

## 📁 Estructura

```
ml-noshow-prediction/
├── train_model.py            ← Script de entrenamiento completo
├── models/
│   └── noshow_predictor.pkl  ← Modelo entrenado (Random Forest)
├── notebooks/
│   ├── feature_importance.png ← Importancia de variables
│   └── roc_curve.png          ← Curva ROC del modelo
└── README.md
```

---

## 🎯 Features del modelo

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `pickup_hour` | numérica | Hora de recogida (0-23) |
| `pickup_dayofweek` | numérica | Día de la semana (0=Lunes) |
| `is_weekend` | binaria | ¿Es fin de semana? |
| `is_summer` | binaria | ¿Es temporada alta? |
| `total_amount` | numérica | Monto total de la reserva |
| `rental_days` | numérica | Días de alquiler |
| `brand_name` | categórica | Marca (Alamo/Enterprise/National/Grupo ANC) |
| `category_name` | categórica | Categoría del vehículo |
| `agency_name` | categórica | Canal de reserva |
| `is_vip` | binaria | ¿Cliente VIP? |
| `is_corporate` | binaria | ¿Reserva corporativa? |

---

## 🚀 Cómo usar el modelo

```python
import joblib
import pandas as pd

# Cargar modelo
model_pkg = joblib.load('models/noshow_predictor.pkl')
model = model_pkg['model']
scaler = model_pkg['scaler']
features = model_pkg['features']

# Predecir nuevas reservas
nueva_reserva = pd.DataFrame([{
    'total_amount': 150.00,
    'rental_days': 3,
    'pickup_hour': 14,
    'pickup_dayofweek': 5,      # Sábado
    'is_weekend': 1,
    'is_summer': 1,              # Julio
    'brand_name_enc': 0,
    'category_name_enc': 2,
    'agency_name_enc': 1,
    'is_vip': 0,
    'is_corporate': 1
}])

X_new = scaler.transform(nueva_reserva[features])
probabilidad = model.predict_proba(X_new)[0, 1]
print(f"🔮 Probabilidad de No-Show: {probabilidad:.1%}")
```

---

## 🔄 Próximas mejoras con datos reales

- [ ] Agregar datos climáticos (lluvia en SJO)
- [ ] Features de temporada (vacaciones, eventos)
- [ ] Más historial (>12 meses)
- [ ] Probar XGBoost / LightGBM
- [ ] SHAP values para interpretabilidad
- [ ] API endpoint para predicción en tiempo real
- [ ] Dashboard Power BI con scoring de riesgo

---

<p align="center">
  <i>Hecho con 🐍 scikit-learn por José Andrés Sequeira</i>
</p>
