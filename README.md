# Proyecto Python: Análisis Exploratorio y Modelado Estadístico Multivariante de Tarifas Aéreas

Este repositorio contiene un proyecto práctico desarrollado en Python utilizando las librerías **Pandas**, **NumPy**, **Seaborn** y **Matplotlib** para realizar un Análisis Exploratorio de Datos (EDA) avanzado sobre las operaciones de una compañía aérea. El script implementa técnicas para mitigar la sobregraficación (*overplotting*) mediante submuestreo aleatorio, evalúa el impacto de servicios a bordo en las tarifas comerciales y analiza correlaciones multivariadas cruzadas (precios de clases, horarios de vuelo y días de la semana) para extraer patrones predictivos de comportamiento de mercado.

---

## Código Python del Proyecto

El programa realiza la limpieza, el cálculo estadístico y la exportación de las visualizaciones multivariadas a archivos locales:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Ingesta de Datos y Diagnóstico Inicial ---
flight = pd.read_csv("flight.csv")
print(flight.head())

# --- 2. Análisis Univariado de Tarifas Básicas (Task 1) ---
coach_mean = np.mean(flight.coach_price)
coach_median = np.median(flight.coach_price)
print(f"Tarifa Turista - Media: {coach_mean} | Mediana: {coach_median}")

sns.histplot(flight.coach_price)
plt.title("Distribución General de Tarifas de Clase Turista")
plt.savefig("hist_tarifas_general.png")
plt.close()

# --- 3. Análisis Segmentado por Duración de Vuelo (Task 2) ---
# Filtrado de costos para vuelos de larga distancia estrictamente iguales a 8 horas
coach_8h_mean = np.mean(flight.coach_price[flight.hours == 8])
coach_8h_median = np.median(flight.coach_price[flight.hours == 8])

sns.histplot(flight.coach_price[flight.hours == 8])
plt.title("Distribución de Tarifas en Vuelos de 8 Horas")
plt.savefig("hist_tarifas_8horas.png")
plt.close()

# --- 4. Análisis de Anomalías y Retrasos (Task 3) ---
# Filtrado de valores extremos o atípicos de demoras en despegues
sns.histplot(flight.delay[flight.delay <= 500])
plt.title("Distribución de Retrasos en Despegues (Filtro <= 500 min)")
plt.savefig("hist_retrasos_filtrado.png")
plt.close()

# --- 5. Mitigación de Overplotting mediante Submuestreo (Task 4) ---
# Extracción de una submuestra probabilística del 10% del dataset original
perc = 0.1
flight_sub = flight.sample(n=int(flight.shape[0] * perc), random_state=42)

# Gráfico de dispersión bivariado (Turista vs Primera Clase) con ajuste suavizado LOWESS
sns.lmplot(x="coach_price", y="firstclass_price", data=flight_sub, line_kws={"color":"black"}, lowess=True)
plt.title("Correlación de Precios: Turista vs Primera Clase")
plt.savefig("scatter_precios_clases.png")
plt.close()

# --- 6. Impacto de Servicios a Bordo en las Tarifas (Task 5) ---
sns.histplot(flight, x="coach_price", hue=flight.inflight_entertainment)
plt.title("Impacto del Entretenimiento a Bordo en el Precio")
plt.savefig("hist_precio_vs_entretenimiento.png")
plt.close()

# --- 7. Análisis Multivariante Avanzado (Tasks 6, 7 y 8) ---
# Relación Horas vs Pasajeros utilizando Jitter y Transparencia (Alpha)
sns.lmplot(x="hours", y="passengers", data=flight_sub, x_jitter=0.25, scatter_kws={"s": 5, "alpha":0.2}, fit_reg=False)
plt.title("Volumen de Pasajeros por Duración de Vuelo")
plt.savefig("scatter_horas_vs_pasajeros.png")
plt.close()

# Precios por Clases segmentados por tipo de jornada (Fin de semana vs Día de semana)
sns.lmplot(x='coach_price', y='firstclass_price', hue='weekend', data=flight_sub, fit_reg=False)
plt.title("Precios de Pasajes según Tipo de Jornada")
plt.savefig("scatter_precios_vs_jornada.png")
plt.close()

# Diagramas de Caja y Bigotes compuestos para análisis cronológico nocturno
sns.boxplot(x="day_of_week", y="coach_price", hue="redeye", data=flight)
plt.title("Distribución de Tarifas por Día y Vuelo Nocturno")
plt.savefig("boxplot_tarifas_cronologicas.png")
plt.close()

```

---

## Resultados y Métricas de Calidad de Datos

La transformación lógica y gráfica de las variables comerciales del negocio permite extraer conclusiones operativas directas:

### 1. Perfil del Comportamiento de Tarifas Aéreas

Al contrastar los diferentes escenarios del dataset mediante uniones y condicionales, se determinan las siguientes métricas en consola:

| Segmento Analizado | Promedio General (Media) | Mediana Central | Comportamiento de la Distribución |
| --- | --- | --- | --- |
| **Clase Turista Total** | ~376.58 USD | ~380.56 USD | Distribución unimodal, simétrica y balanceada. |
| **Vuelos de 8 Horas** | ~431.83 USD | ~437.11 USD | Desplazamiento del centro de masas por costos operativos de larga distancia. |

### 2. Galería de Gráficos e Interpretación de Negocio

Para incluir las visualizaciones en la documentación, guarda los archivos de imagen generados (`.png`) en la raíz del repositorio de GitHub:

#### Análisis de Abandono y Correlaciones de Tarifas

* **Precios por Clases y Jornadas:** Los vuelos en fin de semana muestran un incremento paralelo evidente en ambas categorías tarifarias. El ajuste LOWESS confirma una correlación lineal positiva directa entre el costo turista y primera clase.
* **Distribución Temporal y Vuelos Nocturnos (*Redeye*):** Los diagramas de caja revelan que los vuelos nocturnos mantienen tarifas consistentemente más bajas de lunes a domingo.

| Correlación de Precios de Clases | Caja y Bigotes por Días y Turnos |
| --- | --- |
| <img width="607" height="458" alt="image" src="https://github.com/user-attachments/assets/4f14c633-841e-440c-8ff3-08d902451bdb" /> | <img width="593" height="558" alt="image" src="https://github.com/user-attachments/assets/8067ee0f-96f1-4a39-a7b5-021477780ad6" /> |

---

## Conceptos Técnicos Aplicados

* **Mitigación de la Sobre-representación Gráfica (*Overplotting*)**: Cuando un conjunto de datos contiene cientos de miles de registros, los diagramas de dispersión convencionales acumulan puntos sobre las mismas coordenadas, ocultando la verdadera densidad. Este script resuelve el problema mediante tres capas: submuestreo aleatorio al 10%, control de opacidad (`alpha=0.2`) y la adición de ruido aleatorio en el eje horizontal (`x_jitter=0.25`) para separar visualmente datos discretos repetidos.
* **Ajuste de Regresión Local Suavizada (LOWESS)**: Algoritmo de suavizado no paramétrico que ajusta múltiples modelos de regresión lineal en subregiones locales de los datos. Permite trazar una línea de tendencia flexible que se adapta a la forma geométrica de la nube de puntos sin forzar una estructura matemática rígida de mínimos cuadrados.
* **Agregaciones Multivariadas por Atributos Categóricos (`hue`)**: Parámetro analítico que instruye al motor gráfico a subdividir y mapear de manera paralela una distribución utilizando paletas de colores diferenciadas para cada categoría, permitiendo evaluar el impacto de variables cualitativas (como poseer o no wifi a bordo) sobre métricas cuantitativas continuas de negocio.
