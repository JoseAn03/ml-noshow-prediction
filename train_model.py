#!/usr/bin/env python3
"""
No-Show Prediction Model
Entrena un modelo ML para predecir No-Shows en reservas de vehículos
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os, json, warnings, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score

OUTPUT = "/home/jose-andres/.openclaw/workspace/Proyectos/ml-noshow-prediction"
DATA = "/home/jose-andres/.openclaw/workspace/Proyectos/powerbi-rental-dashboard/datasets"
sns.set_style("whitegrid")

# ─── Cargar ───
reservas = pd.read_csv(f"{DATA}/fact_reservations.csv", parse_dates=['pickup_datetime'])
brands = pd.read_csv(f"{DATA}/dim_brands.csv")
customers = pd.read_csv(f"{DATA}/dim_customers.csv")
categories = pd.read_csv(f"{DATA}/dim_vehicle_categories.csv")
agencies = pd.read_csv(f"{DATA}/dim_agencies.csv")

df = (reservas.merge(brands, on='brand_id')
      .merge(customers, on='customer_id')
      .merge(categories, on='category_id')
      .merge(agencies, on='agency_id'))

print(f"📊 Datos: {len(df)} reservas")
print(f"🚫 No-Show rate: {(df['status']=='no_show').mean()*100:.1f}%")

# ─── Feature Engineering ───
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)
df['is_summer'] = df['pickup_datetime'].dt.month.isin([7,8,12,1]).astype(int)
df['is_vip'] = df['is_vip'].astype(int)
df['is_corporate'] = df['is_corporate'].astype(int)
df['is_no_show'] = (df['status'] == 'no_show').astype(int)

# Codificar categóricas
cat_cols = ['brand_name', 'category_name', 'agency_name']
for col in cat_cols:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col])

# Features
features = [
    'total_amount', 'rental_days', 'pickup_hour', 'pickup_dayofweek',
    'is_weekend', 'is_summer', 'brand_name_enc', 'category_name_enc',
    'agency_name_enc', 'is_vip', 'is_corporate'
]

X = df[features].copy()
y = df['is_no_show'].copy()

print(f"🧮 Features: {len(features)} | Shape: {X.shape}")
print(f"⚖️  Classes: 0={len(y)-y.sum()}, 1={y.sum()} (ratio {y.sum()/len(y):.1%})")

# ─── Train/Test Split ───
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ─── Random Forest ───
print("\n🎯 Entrenando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
y_prob = rf.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"   Accuracy: {acc:.3f}")
print(f"   F1 Score: {f1:.3f}")
print(f"   ROC AUC:  {auc:.3f}")

# ─── Feature Importance ───
imp_df = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
colors = ['#dc2626' if i >= len(imp_df)-3 else '#2563eb' for i in range(len(imp_df))]
plt.barh(imp_df['feature'], imp_df['importance'], color=colors)
plt.title('⭐ Feature Importance - No-Show Predictor', fontsize=14, fontweight='bold')
plt.xlabel('Importancia relativa')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/notebooks/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráfico guardado: feature_importance.png")

# ─── ROC Curve ───
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'Random Forest (AUC={auc:.3f})')
plt.plot([0,1],[0,1], 'k--', alpha=0.3)
plt.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📈 ROC Curve - No-Show Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/notebooks/roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ Gráfico guardado: roc_curve.png")

# ─── Guardar modelo ───
model_pkg = {
    'model': rf,
    'scaler': scaler,
    'features': features,
    'metrics': {'accuracy': float(acc), 'f1': float(f1), 'roc_auc': float(auc)}
}
joblib.dump(model_pkg, f'{OUTPUT}/models/noshow_predictor.pkl')
size_kb = os.path.getsize(f'{OUTPUT}/models/noshow_predictor.pkl') / 1024

print(f"\n✅ Modelo guardado: models/noshow_predictor.pkl ({size_kb:.0f} KB)")
print(f"📊 Resultados finales:")
print(f"   ROC AUC:  {auc:.3f}")
print(f"   F1 Score: {f1:.3f}")
print(f"   Accuracy: {acc:.3f}")

print("\n⭐ Top 5 factores más importantes:")
for _, row in imp_df.tail(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

print("\n✅ Proyecto ML completado!")
