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
