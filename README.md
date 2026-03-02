# Equine Lead: Data-Driven Growth Engine for the Horse Industry (Motor de Crecimiento Basado en Datos para la Industria Equina)
---
Estrategia Dual: Random Forest y K-Means para detectar y capturar clientes potenciales en el sector ecuestre. 

---
## 🎯 Objetivo
---
## 🔧 herramientas y tecnologías
---
## 🗃️ structure_proyect
```

📦 Equinde_lead
│
├── 📁 data
│   ├── 📁 raw
│   │   ├── sessions.parquet
│   │   ├── users.parquet
│   │   ├── datos_merge_users_sessions.csv
│   │   ├── input_lead_preuba_masiva.csv
│   │   └── README.md
│   │
│   └── 📁 processed
│       ├── dataset_completo.csv
│       └── README.md
│
├── 📁 models
│   ├── kmeans_buyer_intersts_k5.joblib
│   ├── scaler_clustering.joblib
│   ├── rf_cutomer_valio_full.joblib
│   ├── rf_prospecting_behavioral.joblib
│   ├── label_encoder_segments.joblib
│   └── README.MD
│   
│
├── 📁 notebooks
│   │ 
│   ├── 📁 synthetic_dataset 
│   │   ├── eda_validacion_dataset_sintetico.ipynb
│   │   ├── settings_dataset_sintetico.ipynb
│   │   └── README
│   │
│   └── 📁 models_production
│       ├── feature_enginnering.ipynb
│       ├── estrategia_dual_training_rf_km.ipynb
│       └── README
│
└── README.md


📦 equine_lead_streamlit
├── 📁 pycache
├── 📁 venv
├── interface.py
├── kmeans_buyer_interests_k5.joblib
├── label_encoder_segments.joblib
├── main.py
├── rf_cutomer.joblib
└── scaler_clustering.joblib

```
---
## 💻Generación de dataset sintético
---
## 📊EDA | Auditoría de datos sintéticos
---
## 🧠Feature Engineering
---
## 🤖 Entrenamiento y evaluación de modelos

              precision    recall  f1-score   support

    Caliente       0.94      0.99      0.96       750
        Frío       0.97      0.91      0.94       750
  Prometedor       0.96      0.83      0.89       750
       Tibio       0.83      0.96      0.89       750

    accuracy                           0.92      3000
   macro avg       0.93      0.92      0.92      3000
weighted avg       0.93      0.92      0.92      3000
---
## 🚀 Deploy del proyecto
---
## 🧑🏽‍💻 Áreas de oportunidad y próximos pasos
