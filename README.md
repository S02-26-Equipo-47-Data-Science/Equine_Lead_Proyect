# Equine Lead: Data-Driven Growth Engine for the Horse Industry (Motor de Crecimiento Basado en Datos para la Industria Equina)
---
Estrategia Dual: Random Forest y K-Means para detectar y capturar clientes potenciales en el sector ecuestre. 

---
## 🎯 Objetivo
Automatizar la calificación y segmentación de prospectos en tiempo real para transformar datos de navegación en oportunidades de venta accionables.

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
│   │   └──  input_lead_preuba_masiva.csv
│   │   
│   │
│   └── 📁 processed
│       └── dataset_completo.csv
│       
│
├── 📁 models
│   ├── kmeans_buyer_intersts_k5.joblib
│   ├── scaler_clustering.joblib
│   ├── rf_cutomer_valio_full.joblib
│   ├── rf_prospecting_behavioral.joblib
│   └──  label_encoder_segments.joblib
│   
│   
│
├── 📁 notebooks
│   │ 
│   ├── 📁 synthetic_dataset 
│   │   ├── eda_validacion_dataset_sintetico.ipynb
│   │   └──  settings_dataset_sintetico.ipynb
│   │   
│   │
│   └── 📁 models_production
│       ├── feature_enginnering.ipynb
│       └──  estrategia_dual_training_rf_km.ipynb
│       
│
└── README.md


📦 equine_lead_streamlit
├── requirements.txt
├── interface.py
├── kmeans_buyer_interests_k5.joblib
├── label_encoder_segments.joblib
├── main.py
├── rf_cutomer.joblib
└── scaler_clustering.joblib

```
---
## 💻Generación de dataset sintético

Usuarios:
```
  user_id  age        location  gender    membership  interes_eventos  \
0      U1   40           Texas  Hombre     community                1   
1      U2   36        New York   Mujer  professional                0   
2      U3   28          Nevada   Mujer     community                1   
3      U4   57  North Carolina   Mujer     community                1   
4      U5   41         Florida   Mujer     community                1   

   interes_accesorios  interes_servicios  interes_caballos  
0                   1                  0                 0  
1                   0                  0                 0  
2                   0                  0                 0  
3                   0                  1                 0  
4                   0                  0                 0  
```
Sesiones:
```
  user_id   device  duration_sec  pages_viewed  redirected  purchase  amount  \
0      U1  desktop    145.474069             4           1     False     0.0   
1      U1   mobile    372.608686             4           1     False     0.0   
2      U1   mobile    304.760670             0           0     False     0.0   
3      U1   mobile    308.940290             1           1     False     0.0   
4      U1  desktop    243.737985             1           0     False     0.0   

           login_time  
0 2025-11-28 20:59:00  
1 2025-12-25 03:51:00  
2 2025-01-29 12:36:00  
3 2025-03-10 06:29:00  
4 2025-04-15 04:51:00  
```
Total sesiones: 47837


---
## 📊EDA | Auditoría de datos sintéticos

```
 #   Column              Non-Null Count  Dtype         
---  ------              --------------  -----         
 0   user_id             47837 non-null  object        
 1   age                 47837 non-null  int64         
 2   location            47837 non-null  object        
 3   gender              47837 non-null  object        
 4   membership          47837 non-null  object        
 5   interes_eventos     47837 non-null  int64         
 6   interes_accesorios  47837 non-null  int64         
 7   interes_servicios   47837 non-null  int64         
 8   interes_caballos    47837 non-null  int64         
 9   device              47837 non-null  object        
 10  duration_sec        47837 non-null  float64       
 11  pages_viewed        47837 non-null  int64         
 12  redirected          47837 non-null  int64         
 13  purchase            47837 non-null  bool          
 14  amount              47837 non-null  float64       
 15  login_time          47837 non-null  datetime64[ns]
 16  login_year          47837 non-null  int32         
 17  login_month         47837 non-null  object        
 18  login_day_of_week   47837 non-null  object        
 19  login_hour          47837 non-null  int32 
```

<img width="1789" height="1490" alt="Image" src="https://github.com/user-attachments/assets/0f98ad6e-19e9-442f-b487-238b34e1fe1b" />

<img width="1018" height="950" alt="Image" src="https://github.com/user-attachments/assets/4153dbf0-42c8-4ed7-8117-ef94df7d36f2" />

Performing Chi-Squared Tests for Independence:
```
Pair: gender vs purchase
  Chi2 Statistic: 2.3090
  P-value: 0.3152
  Interpretation: Fail to reject null hypothesis (variables are independent)

--------------------------------------------------

Pair: membership vs purchase
  Chi2 Statistic: 1378.3670
  P-value: 0.0000
  Interpretation: Reject null hypothesis (variables are dependent)

--------------------------------------------------

Pair: device vs purchase
  Chi2 Statistic: 0.0869
  P-value: 0.9575
  Interpretation: Fail to reject null hypothesis (variables are independent)

--------------------------------------------------

Pair: location vs purchase
  Chi2 Statistic: 22.7420
  P-value: 0.0019
  Interpretation: Reject null hypothesis (variables are dependent)

--------------------------------------------------

Pair: login_month vs purchase
  Chi2 Statistic: 16.2331
  P-value: 0.1327
  Interpretation: Fail to reject null hypothesis (variables are independent)

--------------------------------------------------

Pair: gender vs membership
  Chi2 Statistic: 19.0179
  P-value: 0.0001
  Interpretation: Reject null hypothesis (variables are dependent)

--------------------------------------------------

Pair: device vs membership
  Chi2 Statistic: 1.0891
  P-value: 0.5801
  Interpretation: Fail to reject null hypothesis (variables are independent)

--------------------------------------------------

Pair: login_hour vs purchase
  Chi2 Statistic: 17.6698
  P-value: 0.7751
  Interpretation: Fail to reject null hypothesis (variables are independent)

--------------------------------------------------
```
---
## 🧠Feature Engineering

|index|user\_id|average\_pages\_viewed|average\_duration|sessions\_per\_user|conversion\_rate|dif\_avg\_login\_days|organic\_rate|avg\_ticket|monetary\_log|redirected\_ratio|engagement\_score|age|gender|membership|location|interes\_eventos|interes\_accesorios|interes\_servicios|interes\_caballos|device|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|U1|2\.142857142857143|249\.62772959323524|7|0\.14285714285714285|54\.833333333333336|-3|681\.1767174391864|6\.525288740758331|0\.5714285714285714|534\.9165634140755|40|Hombre|community|Texas|1|1|0|0|desktop|
|1|U10|5\.25|351\.94742575636417|4|0\.25|56\.666666666666664|-3|388\.1071115023171|5\.9638546566220425|0\.25|1847\.7239852209118|51|Mujer|professional|Nevada|1|0|0|1|mobile|
|2|U100|4\.0|366\.40697011581256|8|0\.125|40\.142857142857146|-6|355\.5264646744833|5\.876408471810124|0\.25|1465\.6278804632502|26|Mujer|professional|Florida|0|0|0|1|mobile|
|3|U1000|3\.0|235\.69797760245413|4|0\.0|75\.66666666666667|-2|0\.0|0\.0|0\.5|707\.0939328073624|53|Hombre|community|Kentucky|1|0|0|0|desktop|
|4|U10000|2\.0|438\.7195473628054|2|0\.0|142\.0|-1|0\.0|0\.0|0\.5|877\.4390947256107|27|Hombre|community|Texas|1|1|0|0|desktop|

---
## 🤖 Entrenamiento y evaluación de modelos

### 1. El Modelo “Perfecto” (~99% métricas)
* Variables: comportamiento digital (clics, tiempo de sesión, páginas vistas).
* Por qué ocurre: los datos sintéticos generaron correlaciones muy fuertes entre engagement y conversión, que el Random Forest capturó casi a la perfección.
* Riesgo en deploy: funciona como un sistema basado en reglas. Excelente para leads muy activos, pero vulnerable al overfitting si los datos reales no replican esa consistencia.

```
              precision    recall  f1-score   support

    Caliente       0.94      0.99      0.96       750
        Frío       0.97      0.91      0.94       750
  Prometedor       0.96      0.83      0.89       750
       Tibio       0.83      0.96      0.89       750

    accuracy                           0.92      3000
   macro avg       0.93      0.92      0.92      3000
weighted avg       0.93      0.92      0.92      3000
```
<img width="1002" height="702" alt="Image" src="https://github.com/user-attachments/assets/73cefe25-05f3-45b7-89e9-d6d4c0f10fc1" />

### 2. El Modelo “Realista” (~70% métricas)
- Variables: perfil sociodemográfico (edad, ubicación, etc.).
- Valor: refleja la incertidumbre real del mercado. Un accuracy del 70% es sólido, pues no basta con vivir en una zona exclusiva para comprar un caballo de $50k.
- Utilidad: ayuda en la prospección temprana (antes de entrar al funnel), mientras que el modelo “perfecto” es más útil en la conversión.


```
-- Classification Report for rf_behavior (Behavior Only Model) ---
              precision    recall  f1-score   support

    Caliente       0.67      0.58      0.62       750
        Frío       0.81      0.95      0.87       750
  Prometedor       0.68      0.50      0.58       750
       Tibio       0.72      0.89      0.80       750

    accuracy                           0.73      3000
   macro avg       0.72      0.73      0.72      3000
weighted avg       0.72      0.73      0.72      3000
```
<img width="990" height="490" alt="Image" src="https://github.com/user-attachments/assets/47998a27-36d5-44fb-afd4-6c9976eb94a4" />

### Definición del número de clusters
Se definió el número de clusters a patir del método de codo

<img width="713" height="471" alt="Image" src="https://github.com/user-attachments/assets/39d1cb83-ca03-456d-a336-7303976c6a0d" />

|cluster\_hooks|engagement\_score|average\_duration|sessions\_per\_user|interes\_caballos|interes\_servicios|interes\_eventos|interes\_accesorios|potencial\_monetario|
|---|---|---|---|---|---|---|---|---|
|1|1448\.3271647509753|340\.3375298096587|5\.881614349775785|0\.0|0\.0|0\.12600896860986546|0\.27309417040358747|173\.05613014699026|
|4|1374\.8331343150376|302\.4637734682348|6\.351100811123986|1\.0|0\.19119351100811124|0\.2850521436848204|0\.2711471610660487|221\.80031337397855|
|0|1186\.0436942496228|297\.6186053393858|5\.5872210953346855|0\.0|1\.0|0\.3448275862068966|0\.3093306288032454|163\.9441790207437|
|3|892\.5639910542512|295\.8492850027821|4\.147967479674797|0\.0|0\.0|1\.0|0\.456260162601626|86\.61482917921195|
|2|765\.0076260318524|270\.2135446035371|3\.8566408995080814|0\.0|0\.0|0\.002459592410400562|0\.48278285312719604|70\.85578543817098|

---
## 🚀 Deploy del proyecto
### 🐎 Ecuestre AI: Sistema de Predicción y Segmentación v2.0
Este proyecto es una solución de Machine Learning de punta a punta (End-to-End) diseñada para optimizar la conversión de leads en la industria ecuestre. Utiliza una arquitectura desacoplada con un motor de inferencia en FastAPI y una interfaz de usuario interactiva en Streamlit.

### 🚀 Características Principales
Predicción de Prioridad: Clasificación de leads en Frío, Tibio o Caliente mediante un modelo Random Forest entrenado con 24 variables de comportamiento y demografía.

Segmentación de Buyer Persona: Agrupación automática de usuarios en 5 clusters utilizando K-Means, permitiendo estrategias de marketing personalizadas.

Procesamiento Híbrido: Interfaz para análisis de registros individuales y una sección de carga masiva vía archivos CSV.

Arquitectura Escalable: Separación clara entre la lógica del modelo (Backend) y la visualización (Frontend).

### 🛠️ Stack Tecnológico
Lenguaje: Python 3.9+

Modelado: Scikit-Learn (Random Forest, K-Means, StandardScaler)

Backend: FastAPI (Inferencia de alta velocidad)

Frontend: Streamlit (Panel de control interactivo)

Serialización: Joblib (Carga eficiente de modelos entrenados)

### 📂 Estructura del Proyecto
main.py: API REST que expone el endpoint /predict. Maneja la limpieza de datos, el reindex de columnas y la inferencia.

interface.py: Aplicación web que captura datos del usuario y visualiza los resultados del modelo.

*.joblib: Archivos binarios de los modelos, escaladores y encoders re-entrenados.

Gemini dijo
¡Claro que sí! Un buen README es la cara de tu proyecto. Aquí tienes un resumen estructurado, profesional y técnico que describe exactamente lo que construimos, las herramientas utilizadas y cómo ponerlo en marcha.

### 🐎 Ecuestre AI: Sistema de Predicción y Segmentación v2.0
Este proyecto es una solución de Machine Learning de punta a punta (End-to-End) diseñada para optimizar la conversión de leads en la industria ecuestre. Utiliza una arquitectura desacoplada con un motor de inferencia en FastAPI y una interfaz de usuario interactiva en Streamlit.

### 🚀 Características Principales
Predicción de Prioridad: Clasificación de leads en Frío, Tibio o Caliente mediante un modelo Random Forest entrenado con 24 variables de comportamiento y demografía.

Segmentación de Buyer Persona: Agrupación automática de usuarios en 5 clusters utilizando K-Means, permitiendo estrategias de marketing personalizadas.

Procesamiento Híbrido: Interfaz para análisis de registros individuales y una sección de carga masiva vía archivos CSV.

Arquitectura Escalable: Separación clara entre la lógica del modelo (Backend) y la visualización (Frontend).

### 🛠️ Stack Tecnológico
Lenguaje: Python 3.9+

Modelado: Scikit-Learn (Random Forest, K-Means, StandardScaler)

Backend: FastAPI (Inferencia de alta velocidad)

Frontend: Streamlit (Panel de control interactivo)

Serialización: Joblib (Carga eficiente de modelos entrenados)

### 📂 Estructura del Proyecto
main.py: API REST que expone el endpoint /predict. Maneja la limpieza de datos, el reindex de columnas y la inferencia.

interface.py: Aplicación web que captura datos del usuario y visualiza los resultados del modelo.

*.joblib: Archivos binarios de los modelos, escaladores y encoders re-entrenados.

### 🏁 Instrucciones de Ejecución
1. Preparación del Entorno
```   
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn joblib streamlit requests
```
2. Lanzamiento del Sistema
Se requieren dos terminales activas:

Terminal 1 (Backend):
```
uvicorn main:app --reload
```
Terminal 2 (Frontend):
```
uvicorn main:app --reload
```

### 📊 Variables del Modelo
El sistema procesa variables críticas como:

Comportamiento: Engagement Score, Duración de Sesión, Sesiones Totales, Recencia (Días desde último login).

Demografía: Edad, Ubicación (Florida, Texas, etc.), Género, Dispositivo.

Intereses: Selección binaria de interés en Caballos, Accesorios, Servicios y Eventos.


---
## 🧑🏽‍💻 Áreas de oportunidad y próximos pasos
