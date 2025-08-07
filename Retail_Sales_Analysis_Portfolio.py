# -*- coding: utf-8 -*-

# **Etapa I: Problemática y conjunto de datos**

### Definición y contextualización de la Problemática

The dataset used in this project was obtained from the [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis/data?select=Assignment-1_Data.csv) platform. It contains detailed information on transactions made by a retailer between December 1, 2010, at 08:26:00 and December 9, 2011, at 12:50:00. The minimum available temporal resolution is the hour and minute of each transaction. It is important to note that December 2011 is incomplete, as it only includes transactions up to the 9th, which should be taken into account when performing annual analyses. It is worth noting that exact information on which retailer the data pertains to is not available; similarly, since we have the country where the transaction took place, we assume that the study will primarily focus on a retailer in the United Kingdom.

**Dataset Contents:**

The provided file is named Assignment-1_Data and contains the following attributes:
* BillNo: Invoice number (6 digits) assigned to each transaction. (Type: object)
* Itemname: Product name. (Type: object)
* Quantity: Quantity of each product per transaction. (Type: int64)
* Date: Date and time the transaction was generated. (Type: object)
* Price: Product price. (Type: object)
* CustomerID: Unique number (5 digits) assigned to each customer. (Type: float64)
* Country: Country where each customer resides. (Type: object)

**Dataset Characteristics:**

* File format: .csv
* Number of rows: 522,065
* Number of columns: 7

**Motivation / Objective:**

The selection was made to investigate purchasing behavior patterns in a retail context, focusing on the analysis of trends, seasonality, and relationships between products.

Through exploratory analysis, we seek to understand the structure and quality of the data, identify potential anomalies, and prepare the information for future analysis phases.

Two complementary approaches are envisioned:

Market Basket Analysis: To discover frequently purchased product combinations, which can provide useful insights for commercial strategies such as cross-promotions or personalized offers.

Analysis of purchasing trends and seasonality, segmented by country and by time of year or month, to identify recurring or specific consumption patterns.

**Preliminary Analysis Questions:**

1. Best-Selling Products: What are the most purchased products overall?

2. Country Patterns: Do the most frequently purchased products vary by country or region?

3. Seasonality of Product Purchases: Which products are most frequently purchased at certain times of the year or days of the week?

4. Average Purchase: What is the average spend per transaction?

5. Price Distribution: What price ranges predominate in purchases?

6. Quantity-Price Relationship: How does the quantity purchased vary depending on the product price?

7. Returning Customers: How often do they purchase? What percentage of customers make repeat purchases, and how often?

8. Product Categories: If products are categorized, which categories generate the most sales? If not, can we categorize them?

Confirming these questions can reveal patterns that present opportunities to improve inventory management, plan more effective promotions, and customize offers to specific customer segments, thereby increasing profitability and loyalty.

## Functions and Imports

Next, we group all the necessary imports and functions created throughout the job, regardless of the stage you want to run. We recommend running this part.
"""

!pip install swifter
# # Instalar librerías necesarias
# !pip install dash dash-bootstrap-components pyngrok pandas plotly waitress

# # Agregar tu authtoken
# !ngrok config add-authtoken noneedtobeshown

!pip install countryinfo
!pip install dash
!pip install plotly
!pip install pyngrok
!pip install waitress
!pip install prince==0.7.1

from google.colab import drive
import pandas as pd
import numpy as np
from datetime import date
import holidays
import swifter
import kagglehub
import os
import shutil
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import threading
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from waitress import serve
from pyngrok import ngrok
import time
from scipy import stats
import seaborn as sns
# from countryinfo import CountryInfo
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import prince
from sklearn.preprocessing import OneHotEncoder  # Para crear tabla disyuntiva
from sklearn.compose import ColumnTransformer
import plotly.express as px  # Para visualizaciones interactivas
from prince import MCA
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.metrics import jaccard_score

sns.set_theme(style="whitegrid")

"""### Funciones para preprocesamiento"""

def validar_columna(nombre_columna: str, largo):
  # Validar si existe algun CustomerID con letra
  customer_ids = df[nombre_columna].astype(str).copy()

  # Contar cuántos tienen letras
  cantidad_con_letras = customer_ids.str.contains(r'[A-Za-z]', regex=True).sum()
  print(f"Cantidad de {nombre_columna} con letras: {cantidad_con_letras}")

  cantidad_largo_incorrecto = (customer_ids.str.len() != largo).sum()
  print(f"Cantidad de {nombre_columna} con largo distinto a {largo}: {cantidad_largo_incorrecto}")

def validar_y_convertir_customerid(df: pd.DataFrame, columna: str = 'CustomerID') -> pd.Series:
    """
    Valida una columna tipo float64 que representa IDs numéricos.

    - Verifica si hay decimales (valores no enteros)
    - Verifica si hay letras
    - Muestra la distribución de longitudes
    - Si todos los valores son enteros y sin letras, convierte a int64

    Retorna:
        La columna convertida a int64 si es válida, o la original si no.
    """
    print(f"Tipo actual de la columna '{columna}': {df[columna].dtype}")

    # Verificar si hay decimales
    con_decimales = df[columna] % 1 != 0
    cantidad_con_decimales = con_decimales.sum()
    print(f"Valores con decimales: {cantidad_con_decimales}")

    if cantidad_con_decimales > 0:
        print("\nValores con decimales:")
        print(df.loc[con_decimales, columna])

    # Convertir a string para validar letras y calcular largo
    como_texto = df[columna].astype(str)

    # Verificar si contiene letras
    cantidad_con_letras = como_texto.str.contains(r'[A-Za-z]', regex=True).sum()
    print(f"Valores con letras: {cantidad_con_letras}")

    # Calcular largo del número antes del punto decimal
    largos = como_texto.str.split('.').str[0].str.len()

    # Contar la distribución de longitudes
    distribucion_largos = largos.value_counts().sort_index()
    print("\nDistribución de longitudes de CustomerID:")
    for largo, cantidad in distribucion_largos.items():
        print(f" - Largo {largo}: {cantidad} valores")

    # Verificar si es seguro convertir
    if cantidad_con_decimales > 0 or cantidad_con_letras > 0:
        print("\nNo se puede convertir a entero de forma segura.")
        return df[columna]
    else:
        print("\nTodos los valores son válidos. Se convierte a int64.")
        # Es seguro transformar a int64 ya removimos los nulos
        return df[columna].astype('int64')

def agregar_momentos_temporales(df: pd.DataFrame, col_fecha: str = 'date') -> pd.DataFrame:
    # Extraer día y hora
    df['dia_del_mes'] = df[col_fecha].dt.day
    df['hora'] = df[col_fecha].dt.hour
    df['dia_de_semana'] = df['date'].dt.day_name()

    # Momento del día
    def categorizar_momento_dia(hora):
        if 6 <= hora < 12:
            return 'Mañana'
        elif 12 <= hora < 18:
            return 'Tarde'
        elif 18 <= hora < 22:
            return 'Noche'
        else:
            return 'Madrugada'

    df['momento_del_dia'] = df['hora'].apply(categorizar_momento_dia)

    # Momento del mes
    def categorizar_momento_mes(dia):
        if 1 <= dia <= 10:
            return 'Inicio'
        elif 11 <= dia <= 20:
            return 'Mitad'
        else:
            return 'Fin'

    df['mes'] = df['date'].dt.to_period('M').astype(str)

    df['momento_del_mes'] = df['dia_del_mes'].apply(categorizar_momento_mes)

    # Momento del año (Quarter)
    df['momento_del_año'] = df[col_fecha].dt.to_period('Q').astype(str)

    df[['momento_del_dia',
        'momento_del_mes',
        'momento_del_año',
        'dia_de_semana']] = df[['momento_del_dia',
                                'momento_del_mes',
                                'momento_del_año',
                                'dia_de_semana']].astype('category')

    return df

# def obtener_tld(pais):
#     try:
#         return CountryInfo(pais).info().get('tld', [''])[0]
#     except:
#         return 'No disponible'

# Agregar nuevas categorías para reducir "Otros"
def categorizar_item(item):
    if pd.isna(item):
        return 'Otros'
    item = str(item).lower()

    # Categorías previas
    if any(palabra in item for palabra in ['bread', 'buns', 'rolls']):
        return 'Panificados'
    elif any(palabra in item for palabra in ['milk', 'cream', 'cheese', 'yogurt']):
        return 'Lácteos'
    elif any(palabra in item for palabra in ['apple', 'banana', 'fruit', 'berry', 'grapes']):
        return 'Frutas'
    elif any(palabra in item for palabra in ['soda', 'juice', 'cola', 'water']):
        return 'Bebidas'
    elif any(palabra in item for palabra in ['meat', 'chicken', 'beef', 'sausage']):
        return 'Carnes'
    elif any(palabra in item for palabra in ['chocolate', 'candy', 'bubblegum', 'sweet','marshmallows']):
        return 'Dulces'
    elif any(palabra in item for palabra in ['soap', 'detergent', 'shampoo']):
        return 'Limpieza'
    elif any(palabra in item for palabra in ['light', 'holder', 'bunting', 'lantern', 'votive']):
        return 'Decoración'
    elif any(palabra in item for palabra in ['bag', 'jumbo', 'organiser', 'pouch', 'wallet']):
        return 'Bolsas y Organizadores'
    elif any(palabra in item for palabra in ['cakestand', 'mug', 'cup', 'plate', 'teapot']):
        return 'Cocina y Vajilla'
    elif any(palabra in item for palabra in ['card', 'wrap', 'label', 'tag', 'tape', 'box', 'set']):
        return 'Papelería y Regalos'
    elif any(palabra in item for palabra in ['toy', 'children', 'kids', 'game', 'pencil']):
        return 'Infantil / Juguetes'
    elif any(palabra in item for palabra in ['christmas', 'xmas', 'tree', 'snow', 'santa']):
        return 'Navidad y Festividades'
    elif any(palabra in item for palabra in ['bird', 'heart', 'ornament', 'wicker']):
        return 'Ornamentos y Figuras'
    elif any(palabra in item for palabra in ['cake', 'case', 'baking']):
        return 'Repostería y Accesorios'
    elif any(palabra in item for palabra in ['chalkboard', 'frame', 'sign', 'plaque']):
        return 'Decoración de Pared'
    elif any(palabra in item for palabra in ['slate', 'wooden', 'bamboo']):
        return 'Materiales Naturales'
    elif any(palabra in item for palabra in ['alarm', 'clock', 'watch']):
        return 'Relojes y Despertadores'
    elif any(palabra in item for palabra in ['garden', 'kneeling', 'plant', 'flower']):
        return 'Jardinería y Exterior'
    elif any(palabra in item for palabra in ['ribbon', 'charm', 'craft', 'twine']):
        return 'Cintas y Manualidades'
    elif any(palabra in item for palabra in ['home', 'block', 'words', 'letter']):
        return 'Decoración de Hogar'
    elif any(palabra in item for palabra in ['doormat', 'rack', 'scale', 'peg', 'hanger']):
        return 'Accesorios del Hogar'
    elif any(palabra in item for palabra in ['plaster', 'tin', 'lip balm', 'tissue']):
        return 'Salud y Cuidado Personal'
    elif any(palabra in item for palabra in ['scarf', 'apron', 'sock', 'shirt', 'robe','silk fan']):
        return 'Ropa, Textiles y accesorios'
    elif any(palabra in item for palabra in ['pen', 'notebook', 'calculator', 'timer', 'mouse','board']):
        return 'Tecnología u Oficina'
    elif any(palabra in item for palabra in ['candle', 'scent', 'incense', 'oil', 'burner']):
        return 'Velas y Aromas'
    elif any(palabra in item for palabra in ['retro', 'weigh', 'vintage', 'ivory']):
        return 'Cocina Vintage'
    elif any(palabra in item for palabra in ['drawer cabinet']):
        return 'Muebles'
    else:
        return 'Otros'

"""###Funciones para crear gráficos"""

def formato_miles(x, _):
    return f'{x / 1000:.00f}k'

def graficar_xy(
    data,
    columna_x='mes',
    columna_y='monto_total',
    titulo='Evolución Mensual del Monto Total de Ventas',
    color='green',
    label_y='Ventas Totales (en miles)',
    formato_y_en_miles=False,
    hue=None,
    limites_y=None,
    fontsize=16,
    fontsize_tick=14
):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=data, x=columna_x, y=columna_y, marker='o', color=color, hue= None if hue is None else hue)
    plt.title(titulo, fontsize=fontsize + 2)
    plt.xlabel(columna_x.capitalize(), fontsize=fontsize)
    plt.ylabel(label_y, fontsize=fontsize)

    if formato_y_en_miles:
      plt.gca().yaxis.set_major_formatter(FuncFormatter(formato_miles))

    if limites_y:
      plt.ylim(limites_y)

    plt.tick_params(axis='x', labelsize=fontsize_tick)
    plt.tick_params(axis='y', labelsize=fontsize_tick)

    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def graficar_doble_eje_y(
    data1, x1, y1, label_y1, color1,
    data2, y2, label_y2, color2,
    titulo='Evolución con dos variables',
    formato_y1_en_miles=False,
    formato_y2_en_miles=False,
    limites_y1=None,
    limites_y2=None,
    fontsize=16,
    fontsize_tick=14
):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Eje izquierdo
    ax1.set_title(titulo, fontsize=fontsize + 2)
    sns.lineplot(data=data1, x=x1, y=y1, marker='o', color=color1, ax=ax1)
    ax1.set_xlabel(x1.capitalize(), fontsize=fontsize)
    ax1.set_ylabel(label_y1, fontsize=fontsize, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=fontsize_tick)
    ax1.tick_params(axis='x', labelsize=fontsize_tick, rotation=45)
    ax1.grid(axis='both', linestyle='--', alpha=0.7)

    # Aplicar límites al eje izquierdo si se especifican
    if limites_y1 is not None:
        ax1.set_ylim(limites_y1)

    # Eje derecho
    ax2 = ax1.twinx()
    sns.lineplot(data=data2, x=x1, y=y2, marker='o', color=color2, ax=ax2)
    ax2.set_ylabel(label_y2, fontsize=fontsize, color=color2)
    ax2.tick_params(axis='y', labelsize=fontsize_tick, labelcolor=color2)

    # Aplicar límites al eje derecho si se especifican
    if limites_y2 is not None:
        ax2.set_ylim(limites_y2)

    # Formateadores en miles
    if formato_y1_en_miles:
        ax1.yaxis.set_major_formatter(FuncFormatter(formato_miles))
    if formato_y2_en_miles:
        ax2.yaxis.set_major_formatter(FuncFormatter(formato_miles))

    plt.tight_layout()
    plt.show()

def graficar_evolucion_superpuesta(
    data1,
    data2,
    columna_x='mes',
    columna_y='monto_total',
    titulo='Evolución Mensual de Ventas',
    label_y='Ventas Totales',
    formato_y_en_miles=False,
    color1='green',
    color2='blue',
    label1='Dataset 1',
    label2='Dataset 2',
    limites_y=None
):
    # Crear figura
    plt.figure(figsize=(12, 6))

    # Graficar la primera línea
    sns.lineplot(data=data1, x=columna_x, y=columna_y, marker='o', color=color1, label=label1)

    # Graficar la segunda línea
    sns.lineplot(data=data2, x=columna_x, y=columna_y, marker='o', color=color2, label=label2)

    # Configurar el título y las etiquetas
    plt.title(titulo, fontsize=16)
    plt.xlabel(columna_x.capitalize(), fontsize=14)
    plt.ylabel(label_y, fontsize=14)

    # Formato del eje Y en miles si es necesario
    if formato_y_en_miles:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(formato_miles))

    # Configurar límites del eje Y si se especifican
    if limites_y:
        plt.ylim(limites_y)

    # Personalizar el diseño
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar la leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.show()

def graficar_barras_horizontales_con_valores(
    data,
    columna_x,
    columna_y,
    titulo="Gráfico de barras horizontales",
    xlabel="Cantidad",
    ylabel="Categoría",
    palette="tab10",
    figsize=(15, 6),
    formato_valor="{:,.0f}",
    fontsize_valores=14,
    separacion_valor=0.05,
    fontsize=12
):
    fig, ax = plt.subplots(figsize=figsize)
    barplot = sns.barplot(
        data=data,
        x=columna_x,
        y=columna_y,
        palette=palette,
        ax=ax,
        orient="h"
    )

    # Agregar los valores al final de cada barra
    for bar in barplot.patches:
        ax.text(
            bar.get_width() + separacion_valor,
            bar.get_y() + bar.get_height() / 2,
            formato_valor.format(bar.get_width()),
            ha='left',
            va='center',
            fontsize=fontsize_valores
        )

    # Títulos y etiquetas con tamaño de fuente personalizado
    ax.set_title(titulo, fontsize=fontsize + 2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # Cambiar tamaño de fuente de los ticks
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.show()

def graficar_dispersión_interactiva(
    data,
    x,
    y,
    color=None,
    hover_cols=None,
    titulo="Gráfico interactivo: variables cruzadas",
    etiquetas=None,
    ancho=800,
    alto=500,
    plantilla="plotly_white",
    animation_frame=None
):
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        hover_data=hover_cols,
        title=titulo,
        labels=etiquetas,
        width=ancho,
        height=alto,
        animation_frame=animation_frame
    )
    fig.update_layout(template=plantilla)
    fig.show()

def graficar_heatmap(
    data,
    fila,
    columna,
    valor,
    aggfunc="sum",
    orden_filas=None,
    orden_columnas=None,
    titulo="Mapa de calor",
    xlabel=None,
    ylabel=None,
    figsize=(10, 6),
    cmap="YlGnBu",
    fmt=".0f",
    annot=True
):
    # Crear tabla cruzada
    tabla = data.pivot_table(index=fila, columns=columna, values=valor, aggfunc=aggfunc)

    # Reordenar si es necesario
    if orden_filas:
        tabla = tabla.reindex(index=orden_filas)
    if orden_columnas:
        tabla = tabla.reindex(columns=orden_columnas)

    # Crear gráfico
    plt.figure(figsize=figsize)
    sns.heatmap(tabla, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(titulo)
    plt.xlabel(xlabel if xlabel else columna)
    plt.ylabel(ylabel if ylabel else fila)
    plt.tight_layout()
    plt.show()

"""### Funciones de medidas de dispersión"""

def visualizar_distribucion(datos, columna):
    """Visualiza la distribución de una variable con histograma, boxplot y QQ plot."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Histograma con curva de densidad
    sns.histplot(datos[columna], kde=True, ax=axes[0])
    axes[0].axvline(datos[columna].mean(), color='red', linestyle='--', label='Media')
    axes[0].axvline(datos[columna].median(), color='green', linestyle='-.', label='Mediana')
    axes[0].set_title(f'Histograma de {columna}')
    axes[0].legend()

    # Boxplot
    sns.boxplot(x=datos[columna], ax=axes[1])
    axes[1].set_title(f'Boxplot de {columna}')

    # Mostrar estadísticas resumidas
    print(f"\nEstadísticas resumidas para {columna}:")
    print(datos[columna].describe())

    # Mostrar medidas de forma
    print(f"\nAsimetría: {datos[columna].skew():.4f}")
    print(f"Curtosis: {datos[columna].kurtosis():.4f}")

def visualizar_hist_and_qq(df, n_cols=2):
    """Visualiza la distribución de una variable con histograma, boxplot y QQ plot."""
    # Crear un grid de gráficos
    fig, axes = plt.subplots(n_cols, 2, figsize=(12, 10))

    # Iterar sobre cada conjunto de datos
    for i, columna in enumerate(df.columns):
        datos = df[columna]

        # Histograma con curva normal superpuesta
        sns.histplot(datos, kde=True, ax=axes[i, 0])

        # Añadir curva normal teórica
        x = np.linspace(datos.min(), datos.max(), 100)
        media = datos.mean()
        std = datos.std()
        y = stats.norm.pdf(x, media, std)
        axes[i, 0].plot(x, y * len(datos) * (datos.max() - datos.min()) / 10,
                        'r--', linewidth=2)

        axes[i, 0].set_title(f'Histograma - {columna}')

        # Gráfico Q-Q
        stats.probplot(datos, plot=axes[i, 1])
        axes[i, 1].set_title(f'Gráfico Q-Q - {columna}')

    plt.tight_layout()
    plt.show()

def calcular_medidas_posicion(datos):
    """Calcula y muestra las principales medidas de posición."""
    resultados = pd.DataFrame({
        'Media': datos.mean(),
        'Mediana': datos.median(),
        # 'Moda': datos.mode().iloc[0],  # Toma el primer valor si hay múltiples modas
        'Q1 (25%)': datos.quantile(0.25),
        'Q3 (75%)': datos.quantile(0.75),
        'Mínimo': datos.min(),
        'Máximo': datos.max()
    })

    return resultados

def calcular_medidas_dispersion(datos):
    """Calcula y muestra las principales medidas de dispersión."""
    resultados = pd.DataFrame({
        'Rango': datos.max() - datos.min(),
        'Varianza': datos.var(),
        'Desviación Estándar': datos.std(),
        'Coef. de Variación (%)': (datos.std() / datos.mean()) * 100,
        'IQR': datos.quantile(0.75) - datos.quantile(0.25),
        #'MAD': datos.mad()  # Desviación absoluta media
    })

    return resultados

def calcular_medidas_forma(datos):
    """Calcula y muestra las principales medidas de forma."""
    resultados = pd.DataFrame({
        'Asimetría': datos.skew(),
        'Curtosis': datos.kurtosis()
    })

    return resultados

def calcular_estadisticos(datos):
  print("Medidas de Posición:")
  print(calcular_medidas_posicion(datos))
  print("\nMedidas de Dispersión:")
  print(calcular_medidas_dispersion(datos))
  print("\nMedidas de Forma:")
  print(calcular_medidas_forma(datos))

"""###Funciones de detección y eliminación de outliers"""

def detectar_outliers_iqr(datos, columna, threshold=1.5):
    """
    Detecta valores atípicos utilizando el método IQR

    Returns:
        DataFrame con los outliers identificados
    """
    Q1 = datos[columna].quantile(0.25)
    Q3 = datos[columna].quantile(0.75)
    IQR = Q3 - Q1

    # 1.5 Una desvición y media
    limite_inferior = Q1 - threshold * IQR
    limite_superior = Q3 + threshold * IQR

    outliers = datos[(datos[columna] < limite_inferior) |
                      (datos[columna] > limite_superior)]

    print(f"Método IQR para {columna}:")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Límite inferior: {limite_inferior:.2f}")
    print(f"Límite superior: {limite_superior:.2f}")

    print(f"Número de outliers encontrados: {len(outliers)}")
    return outliers

def detectar_outliers_zscore(datos, columna, threshold=3):
    """
    Detecta valores atípicos utilizando el método Z-score.

    Returns:
        DataFrame con los outliers identificados
    """
    z_scores = stats.zscore(datos[columna])
    outliers = datos[abs(z_scores) > threshold]

    print(f"Método Z-score para {columna}:")
    print(f"Valores con |z| > {threshold} son considerados outliers")
    print(f"Número de outliers encontrados: {len(outliers)}")
    return outliers

def z_score_filter(datos, columna, threshold=3):
    z_scores = stats.zscore(datos[columna])
    return datos[abs(z_scores) <= threshold]

"""###Funciones de test de normalidad"""

# Función para aplicar múltiples tests de normalidad
def test_normalidad(datos, nombre=None):
    """
    Aplica múltiples tests de normalidad a un conjunto de datos
    y devuelve los resultados en un DataFrame.
    """
    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(datos)

    # Anderson-Darling test
    anderson_test = stats.anderson(datos, dist='norm')

    # Kolmogorov-Smirnov test
    ks_test = stats.kstest(datos, 'norm', args=(datos.mean(), datos.std()))

    # Crear DataFrame con resultados
    resultados = pd.DataFrame({
        'Test': ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
        'Estadístico': [shapiro_test[0], ks_test[0]],
        'p-valor': [shapiro_test[1], ks_test[1]],
        'Normalidad (α=0.05)': [
            'Aceptada' if shapiro_test[1] > 0.05 else 'Rechazada',
            'Aceptada' if ks_test[1] > 0.05 else 'Rechazada'
        ]
    })

    # Añadir resultados de Anderson-Darling
    # (este test tiene múltiples valores críticos)
    ad_result = 'Rechazada'
    for i, nivel in enumerate([15, 10, 5, 2.5, 1]):
        if anderson_test[0] < anderson_test[1][i]:
            ad_result = f'Aceptada (α={nivel}%)'
            break

    ad_row = pd.DataFrame({
        'Test': ['Anderson-Darling'],
        'Estadístico': [anderson_test[0]],
        'p-valor': [None],  # Anderson-Darling no reporta p-valor directamente
        'Normalidad (α=0.05)': [ad_result]
    })

    resultados = pd.concat([resultados, ad_row], ignore_index=True)

    if nombre:
        print(f"Resultados para: {nombre}")

    return resultados

"""###Funciones análisis univariado y bivariado"""

# Análisis univariado para cada variable categórica
def analisis_univariado(df, var):
    """Realiza un análisis univariado completo para una variable categórica"""
    print(f"Análisis de la variable: {var}")

    # Valores nulos
    n_missing = df[var].isna().sum()
    pct_missing = 100 * n_missing / len(df)
    print(f"Valores nulos: {n_missing} ({pct_missing:.2f}%)")

    # Distribución de frecuencias
    value_counts = df[var].value_counts()
    print("\nDistribución de frecuencias:")

    # Crear tabla con frecuencias absolutas y relativas
    freq_table = pd.DataFrame({
        'Frecuencia': value_counts,
        'Porcentaje (%)': 100 * value_counts / value_counts.sum()
    })
    print(freq_table)

    # Visualización
    plt.figure(figsize=(14, 10))

    # Gráfico de barras
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x=var, order=value_counts.index)
    plt.title(f'Frecuencia de {var}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Frecuencia')

    # Gráfico de barras normalizado
    plt.subplot(2, 2, 2)
    sns.barplot(x=freq_table.index, y='Porcentaje (%)', data=freq_table.reset_index())
    plt.title(f'Distribución porcentual de {var}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Porcentaje (%)')

    # Gráfico de pastel
    plt.subplot(2, 2, 3)
    plt.pie(freq_table['Frecuencia'], labels=freq_table.index, autopct='%1.1f%%')
    plt.title(f'Proporción de {var}')

    # Estadísticas de resumen (si es ordinal o se puede interpretar numéricamente)
    if df[var].dtype in ['int64', 'float64']:
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, y=var)
        plt.title(f'Boxplot de {var}')

        # Algunas estadísticas adicionales
        print("\nEstadísticas descriptivas:")
        print(df[var].describe())

    plt.tight_layout()
    plt.show()
    print("\n" + "="*50 + "\n")

# Análisis bivariado: relación entre variables categóricas
def analisis_bivariado(df, var1, var2, target=None, should_graph=True):
    """Analiza la relación entre dos variables categóricas"""
    print(f"Análisis bivariado: {var1} vs {var2}")

    # Tabla de contingencia
    cont_table = pd.crosstab(df[var1], df[var2], margins=True, margins_name="Total")
    print("Tabla de contingencia (frecuencias absolutas):")
    print(cont_table)

    # Tabla de contingencia normalizada (porcentajes por fila)
    cont_table_norm = pd.crosstab(df[var1], df[var2], normalize='index')
    print("\nTabla de contingencia (porcentajes por fila):")
    print(cont_table_norm.round(3) * 100)

    # Prueba Chi-cuadrado
    # Tabla sin los márgenes para chi2
    table_no_margins = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(table_no_margins)
    print(f"\nPrueba Chi-cuadrado:")
    print(f"Chi2: {chi2:.4f}")
    print(f"Valor p: {p:.4f}")
    print(f"Grados de libertad: {dof}")
    if p < 0.05:
        print("Las variables están relacionadas (p < 0.05)")
    else:
        print("No hay evidencia suficiente de relación (p >= 0.05)")

    # Coeficiente V de Cramér (mide la fuerza de asociación)
    n = table_no_margins.sum().sum()
    phi = np.sqrt(chi2 / (n * min(table_no_margins.shape) - 1))
    print(f"V de Cramér: {phi:.4f}")

    # Interpretar la fuerza de la relación
    if phi < 0.1:
        print("Relación muy débil")
    elif phi < 0.3:
        print("Relación débil")
    elif phi < 0.5:
        print("Relación moderada")
    else:
        print("Relación fuerte")

    # Visualización
    if should_graph:
      plt.figure(figsize=(14, 10))

      # Gráfico de barras apiladas
      plt.subplot(2, 2, 1)
      pd.crosstab(df[var1], df[var2]).plot(kind='bar', stacked=True, ax=plt.gca())
      plt.title(f'Gráfico de barras apiladas: {var1} vs {var2}')
      plt.xlabel(var1)
      plt.ylabel('Frecuencia')
      plt.legend(title=var2)
      plt.xticks(rotation=45, ha='right')

      # Gráfico de calor
      plt.subplot(2, 2, 2)
      sns.heatmap(pd.crosstab(df[var1], df[var2]), annot=True, fmt='d', cmap='viridis')
      plt.title(f'Mapa de calor: {var1} vs {var2}')

      # Gráfico de mosaico
      plt.subplot(2, 2, 3)
      mosaic_df = df[[var1, var2]].dropna()
      mosaic_data = {(i, j): len(mosaic_df[(mosaic_df[var1] == i) & (mosaic_df[var2] == j)])
                    for i in mosaic_df[var1].unique() for j in mosaic_df[var2].unique()}
      mosaic(mosaic_data, ax=plt.gca())
      plt.title(f'Gráfico de mosaico: {var1} vs {var2}')

      # Si hay una variable objetivo, mostrar la relación con ambas variables
      if target is not None and target in df.columns:
          plt.subplot(2, 2, 4)
          cross_target = pd.crosstab([df[var1], df[var2]], df[target], normalize='index')
          if cross_target.shape[1] == 2:  # Si el target es binario
              # Tomar solo una columna para un mapa de calor más claro
              target_col = cross_target.columns[1]
              cross_target_pivot = cross_target[target_col].unstack()
              sns.heatmap(cross_target_pivot, annot=True, fmt='.2%', cmap='RdYlGn')
              plt.title(f'Proporción de {target}={target_col} por {var1} y {var2}')
          else:
              plt.text(0.5, 0.5, "Visualización disponible solo para targets binarios",
                      horizontalalignment='center', verticalalignment='center')

      plt.tight_layout()
      plt.show()

"""### Funciones de series de tiempo"""

def test_stationarity(timeseries, window=12, title='', should_graph=True):
    # Cálculo de estadísticas móviles
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()

    # Gráfica de la serie y sus estadísticas móviles
    if (should_graph):
      plt.figure(figsize=(10, 6))
      plt.plot(timeseries, 'b-', label='Original')
      plt.plot(rolling_mean, 'r-', label=f'Media Móvil (ventana={window})')
      plt.plot(rolling_std, 'g-', label=f'Desv. Est. Móvil (ventana={window})')
      plt.title(f'Análisis de Estacionariedad: {title}', fontsize=18)
      plt.legend(loc='best', fontsize=14)
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      #plt.savefig(f'estacionariedad_{title.replace(" ", "_").lower()}.png', dpi=100)

    # Test de Dickey-Fuller Aumentado
    print(f"\nTest de Dickey-Fuller Aumentado para {title}:")
    adf_test = adfuller(timeseries.dropna(), autolag='AIC')
    adf_output = pd.Series(adf_test[0:4], index=['Estadístico ADF', 'p-valor', '# de Lags', '# de Observaciones'])
    for key, value in adf_test[4].items():
        adf_output[f'Valor Crítico ({key})'] = value
    print(adf_output)

    if adf_test[1] <= 0.05:
        print("Conclusión: La serie es estacionaria (rechaza la hipótesis nula)")
    else:
        print("Conclusión: La serie no es estacionaria (no rechaza la hipótesis nula)")

    # Test KPSS
    print(f"\nTest KPSS para {title}:")
    kpss_test = kpss(timeseries.dropna(), regression='c')
    kpss_output = pd.Series(kpss_test[0:3], index=['Estadístico KPSS', 'p-valor', '# de Lags'])
    for key, value in kpss_test[3].items():
        kpss_output[f'Valor Crítico ({key})'] = value
    print(kpss_output)

    if kpss_test[1] <= 0.05:
        print("Conclusión: La serie no es estacionaria (rechaza la hipótesis nula)")
    else:
        print("Conclusión: La serie es estacionaria (no rechaza la hipótesis nula)")

"""### Funciones de correlación"""

def varianza_correlacion(df, var_1, var_2):
  # Varianza
  print(f"Varianza de {var_1}:", np.var(df_sales_date[var_1], ddof=1))
  print(f"Varianza de {var_2}:", np.var(df_sales_date[var_2], ddof=1))
  print("\n")

  # Matriz de covarianza
  cov_matrix = df[[var_1, var_2]].cov()

  # Heatmap de la matriz de covarianza
  plt.figure(figsize=(6, 4))
  sns.heatmap(
    cov_matrix,
    annot=True,
    cmap='PuBu',
    center=0,
    fmt=".2f",
    linewidths=0.5
  )
  plt.title("Matriz de covarianza", fontsize=12)
  plt.show()

"""### Fuciones de correspondencia"""

def ejecutar_mca(df_mca, titulo='Mapa de Categorías'):
    """
    Realiza un Análisis de Correspondencia Múltiple (MCA) sobre las columnas categóricas especificadas
    y muestra el gráfico de coordenadas principales.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        columnas_categoricas (list): Lista con los nombres de las columnas categóricas.
        titulo (str): Título para el gráfico.

    Retorna:
        dict: Contiene eigenvalues, explained_inertia, coordenadas de categorías e individuos.
    """

    # Inicializar modelo MCA
    mca = prince.MCA(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )

    # Ajustar modelo
    mca_resultado = mca.fit(df_mca)

    # Obtener métricas y coordenadas
    eigenvalues = mca.eigenvalues_
    explained_inertia = mca.explained_inertia_
    coord_categorias = mca.column_coordinates(df_mca)
    coord_individuos = mca.row_coordinates(df_mca)

    # Graficar
    ax = mca.plot_coordinates(
        X=df_mca,
        ax=None,
        figsize=(8, 6),
        show_row_points=True,
        row_points_size=10,
        show_row_labels=False,
        show_column_points=True,
        column_points_size=30,
        show_column_labels=True,
        legend_n_cols=1
    )
    plt.title(titulo)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Retornar resultados
    return {
        'eigenvalues': eigenvalues,
        'explained_inertia': explained_inertia,
        'coord_categorias': coord_categorias,
        'coord_individuos': coord_individuos
    }

"""### Funciones similitud"""

# Función de recomendación simple
def recomendar_similar(target, similarity_matrix, method_name, top_n=2):
    # Encontrar usuarios más similares
    user_similarities = similarity_matrix.loc[target].drop(target)
    most_similar_categorias = user_similarities.nlargest(top_n)

    return most_similar_categorias

def comparacion_metodos_similitud(categoria_objetivo, manhattan_df, euclidean_df, jaccard_df, top_n=2):
  similares_manhattan = recomendar_similar(categoria_objetivo, manhattan_df, "Manhattan", top_n)
  similares_euclidiana = recomendar_similar(categoria_objetivo, euclidean_df, "Euclidiana", top_n)
  similares_jaccard = recomendar_similar(categoria_objetivo, jaccard_df, "Jaccard", top_n)

  # Comparación de resultados
  print("\n" + "="*60)
  print("COMPARACIÓN DE MÉTODOS DE SIMILITUD")
  print("="*60)

  comparison_data = {
      'Método': ['Manhattan', 'Euclidiana', 'Jaccard'],
      f'Categoria más similar a {categoria_objetivo}': [
          similares_manhattan.index[0],
          similares_euclidiana.index[0],
          similares_jaccard.index[0]
      ],
      'Puntuación de similitud': [
          similares_manhattan.iloc[0],
          similares_euclidiana.iloc[0],
          similares_jaccard.iloc[0]
      ]
  }

  comparison_df = pd.DataFrame(comparison_data)
  print(comparison_df.round(3))

"""### Data Mockeada para testing"""

# Se crea un mock de DataFrame para testings
df_mock = pd.DataFrame({
    'bill_no': ['123456', '123457', 'ABC123', None, '123459'],
    'item_name': ['Chair', 'Table', '', 'Lamp', 'Chair'],
    'quantity': [5, -3, 2, 'three', None],  # número negativo y texto no numérico
    'date': ['2010-12-01 08:26:00', 'not_a_date', '2011-05-01', '', None],  # formato incorrecto y nulo
    'price': [10.5, -20.0, 15.0, 'free', np.nan],  # precio negativo y texto
    'customer_ID': ['12345', 'ABCDE', '67890', '12345', None],  # ID inválido (texto) y duplicado
    'country': ['United Kingdom', '', 'France', 'Germany', None]  # valores vacíos o nulos
})

"""## Descripción y calidad de los datos

**Preguntas sobre los datos**
* **Completitud:** ¿Faltan datos?
* **Validez:** ¿Cumplen con los formatos esperados?
* **Precisión:** ¿Son correctos y fiables?
* **Consistencia:** ¿Coinciden los datos entre fuentes?
* **Relevancia:** ¿Los datos son pertinentes para tener resultados concluyentes del problema en estudio?
* **Unicidad:** ¿Ausencia de registros duplicados?
* **Integridad:** ¿Datos completos y sin corrupción?

## Guía de lectura

En las etapas siguientes del proyecto se continuará con un desarrollo estructurado del análisis. En la Etapa II se abordará la limpieza y preparación de los datos, así como la justificación de la calidad de estos. En la Etapa III, se realizará la visualización y el análisis para responder a las preguntas planteadas en esta etapa inicial. Finalmente, en la Etapa IV se presentarán y discutirán los hallazgos obtenidos del análisis realizado.

# Etapa II: Limpieza y preprocesamiento de los datos

### Carga del Dataframe
"""

# Se importa drive para utilizarlo en la lectura del archivo
drive.mount('/content/drive')

# Se importa pandas previamente para leer el archivo desde el Drive.
# El dataset a trabajar debe estar almacenado en /content/drive/My Drive/Notebooks/ con el nombre AssignmentData.csv

ruta = '/content/drive/My Drive/Notebooks/'
nombre_archivo = 'AssignmentData.csv'
df_market = pd.read_csv(ruta + nombre_archivo, on_bad_lines=lambda line: print(f"Saltando línea: {line}"), engine='python',sep=';')
df_market.head()

df = df_market.copy() # Generamos una copia del dataframe, que será sobre la que trabajaremos, para mantener la original de respaldo

df

"""### Inspección de los datos

- Es necesario estandarizar formatos?
- Los datos cumplen con los rangos esperados?
- Hay algunas columas que son del tipo numérica pero deberían tratarse como string? Ej: CustomerId
- Existen duplicados?
- Hay datos faltantes?
"""

df.shape

df.head()

"""### Renombrar columnas"""

df.columns

df = df.rename(columns={'BillNo': 'bill_no',
                        'Itemname': 'item_name',
                        'Quantity': 'quantity',
                        'Date': 'date',
                        'Price': 'price',
                        'CustomerID': 'customer_id',
                        'Country': 'country'})
df.columns

"""### Tipo de dato

Todas las columnas tienen su tipo de dato correcto?
"""

df.info()

"""Se convierte la columna Price a float porque se detecta que el separador de decimales es "," y no lo está tomando como numérico.


"""

df['price'] = df['price'].astype(str).str.replace(',', '.').astype(float)

"""Se convierte la columna date a tipo datetime, se podría evaluar la creación de columnas hora, día de la semana y día del mes para el análisis posterior."""

# Validar fechas
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

"""Chequeamos la existencia de ninguna fecha futura"""

print(f"Fecha min {df.date.min()}, fecha max: {df.date.max()}.")

cantidad_nat = df['date'].isna().sum()
print(f"Cantidad de fechas no válidas (NaT): {cantidad_nat}")

"""Resolución temporal más pequeña disponible"""

# Verificar si hay horas, minutos o segundos distintos de 00:00:00
resoluciones = df['date'].dt.time.unique()

if len(resoluciones) == 1 and resoluciones[0].strftime('%H:%M:%S') == '00:00:00':
    print("La resolución temporal más pequeña disponible en el dataset es un día completo (fecha sin hora).")
else:
    print("El dataset contiene información horaria. Se recomienda evaluar la resolución mínima entre registros.")

"""Investigamos sobre la composicion de customer_id para ver si es posible convertirlo a int"""

df['customer_id'] = validar_y_convertir_customerid(df, columna='customer_id')

"""Validamos si los datos en la columna bill_no contienen letras o si los datos son de largo distinto"""

validar_columna("bill_no", 6)

"""Investigamos sobre la existencia de bill_no con letra, vemos que podría ser una transacción negativa, una devolución contable o una corrección interna"""

billnos = df['bill_no'].astype(str)

# Filtrar las filas que contienen letras en bill_no
filas_con_letras = df[billnos.str.contains(r'[A-Za-z]', regex=True)]
filas_con_letras.head()

# Eliminar filas que contienen letras en el bill_no
df = df[~df['bill_no'].str.contains(r'[A-Za-z]', regex=True)]

print(df['bill_no'].str.contains(r'[A-Za-z]', regex=True).sum())

print(df['item_name'].str.contains('Adjust bad debt', case=False, na=False).sum())

"""Como hemos removido los valores con letras, ahora es seguro transformar a Int"""

billnos = df['bill_no']
# Chequeamos si hay punto o coma en alguno de los valores
contiene_punto_o_coma = billnos.str.contains(r'[.,]', regex=True)

# Mostramos la cantidad de casos
print(f"Cantidad de bill_no con punto o coma: {contiene_punto_o_coma.sum()}")

# Chequeo de nulos o strings vacíos/desconocidos
vacíos_o_nulos = billnos.isna() | (billnos.str.strip() == "")

print(f"Cantidad de bill_no vacíos o nulos: {vacíos_o_nulos.sum()}")

"""Es seguro transformar a int"""

df['bill_no'] = pd.to_numeric(df['bill_no'], errors='coerce').astype('int64')

"""Chequeamos que quedaron correctos los tipos de datos."""

# Chequeo de tipos post conversion
df.info()

df.head()

"""Antes de Operar con el tipo para la columna Country, válidamos el % existente de datos para cada país. Vemos que la mayor cantidad de datos que poseemos es de United Kingdom, por lo que continuaremos nuestros estudios sobre este sub conjunto. No hay suficiente información sobre los otros países para realizar una comparativa"""

porcentaje_paises = df['country'].value_counts(normalize=True) * 100
print(porcentaje_paises)

"""No modificamos el tipo de ItemName y CustomerID en esta fase porque vemos la existencia de nulos, a explorar en la siguiente sección.

### Valores nulos

Proceso de estrategia para imputar valores nulos
"""

# Cantidad de nulos por columna
df.isnull().sum()

# Porcentaje de valores nulos (faltantes) en cada columna
round((df.isnull().sum()/len(df)), 2)*100

"""Se procede a chequear los elementos donde item_name sea nulos, en este caso procederemos a la eliminación de ellos, porque al no tener el item_name y el Price no nos aporta información relevante."""

df_nulos_itemName = df[df.item_name.isnull()]
df_nulos_itemName.describe()

df = df.dropna(subset=['item_name'])

"""Transformación de Itemname a string ahora es seguro"""

df['item_name'] = df['item_name'].astype('string')

"""Se procede a chequear los elementos donde CustomerId sea nulos, en este caso procederemos a la imputación con el 99999 (previo chequeo de su no existencia),
para que quede registro de la sustitución realizada. Dado que contamos con información en el resto de las columnas decidimos imputar un valor.
"""

df_nulos_customerID = df[df.customer_id.isnull()]
df_nulos_customerID.describe()

"""Chequeo de no existencia CustomerID 99999 e Imputación CustomerID Nulos a 99999"""

existe = 99999 in df['customer_id'].values

print(f"¿Existe el customer_id {99999}? {existe}")
df.loc[df['customer_id'].isna(), 'customer_id'] = 99999

df.isnull().sum()

print(f"Cantidad de customer_id menor que cero: {df[df['customer_id'] <= 0].shape[0]}")
validar_columna("customer_id", 5)

df['customer_id'] = validar_y_convertir_customerid(df, columna='customer_id')

"""Solo Country queda como object, de existir mas datos lo hubiesemos trabajado como category"""

# Chequeo de tipos post conversion
df.info()

"""### Duplicados

Chequeamos si hay duplicados y los analizamos para evaluar cuál es la mejor estrategia.
"""

# Tipo de datos
df.head()

"""Buscamos si hay duplicados e inspeccionamos ejemplos para tomar una decisión de qué estrategia tomar."""

df.duplicated().sum()

df[df.duplicated()]

# Buscamos si hay duplicados y cuantos son
print(f"Hay {df.duplicated().sum()} duplicados.")
print(f"Estos representan un total de {round(df[df.duplicated(keep=False)].shape[0]/df.shape[0], 2)*100} %")

"""Seleccionamos un ejemplo para poder entender el porqué de los duplicados."""

# Buscamos por bill number 536412
bill_number = 536412
df[df.duplicated()].loc[df["bill_no"] == bill_number].sort_values(by="item_name").head(20)

"""Dado que no estamos seguros de que los duplicados sean realmente duplicados, planteamos la posibilidad de hacer una suma de las cantidades agrupando por el resto de las variables porque puede que se pasen los items por separado a la hora de hacer la factura.

Antes de hacerlo vamos a analizar si cambia la distribución de la variable Quantity cuando se realiza la suma.
"""

df.quantity.describe()

df[['item_name', 'customer_id']] = df[['item_name', 'customer_id']].fillna('Unknown')
df_grouped = df.groupby(['bill_no', 'item_name', 'date', 'price', 'customer_id','country'])['quantity'].sum().reset_index()
df_grouped.quantity.describe()

#Calculamos media, varianza y desviación de Quantity con numpy
mean_quantity = np.mean(df['quantity'])
var_quantity = np.var(df['quantity'])
std_quantity = np.std(df['quantity'])

print(f"Media de quantity: {round(mean_quantity,2)}")
print(f"Varianza de quantity: {round(var_quantity,2)}")
print(f"Desviación estándar de quantity: {round(std_quantity,2)}")

#Calculamos media, varianza y desviación despues de agrupados de Quantity con numpy
mean_quantity_group = np.mean(df_grouped['quantity'])
var_quantity_group = np.var(df_grouped['quantity'])
std_quantity_group = np.std(df_grouped['quantity'])

print(f"Media de quantity: {round(mean_quantity_group,2)}")
print(f"Varianza de quantity: {round(var_quantity_group,2)}")
print(f"Desviación estándar de quantity: {round(std_quantity_group,2)}")

"""Vemos que no cambia mucho la distribución de los datos pero detectamos que hay muchos valores negativos que hay que revisar.

Miramos como quedan algunos ejemplos luego de la suma de las cantidades
"""

# Buscamos por bill number 536412, 536412, 536589
bill_number = 536412

# Datos antes de la agrupacion
df[df["bill_no"] == bill_number].sort_values(by="item_name").head(20)

# Datos posterior a la agrupacion
df_grouped[df_grouped["bill_no"] == bill_number].sort_values(by="item_name").head(20)

"""### Análisis de valores extraños / Consistencia

Durante la exploración detectamos valores con cantidad negativos, y nombre de items no correspondientes a un producto. Dando indicios a no representar una venta.
"""

df_grouped[['price', 'quantity']].describe()

# Filtrado de ventas con cantidades negativas
df_grouped[df_grouped.quantity<0].sort_values(by='bill_no')

"""De estos valores negativos podemos inferir que son devoluciones, como tienen CustomerID 99999 (que correspondia al nulo previo imputacion) y no tiene precio."""

df_grouped[(df_grouped['quantity'] < 0) & (df_grouped['price'] <= 0)]

# Chequeo de existencia de ventas con cantidad negativa y valor positivo de venta
df_grouped[(df_grouped['quantity'] < 0) & (df_grouped['price'] > 0)]

# Observamos los Itemname para las transacciones con cantidad negativa
df_grouped[df_grouped['quantity'] < 0].item_name.value_counts()

"""Posterior análisis decidimos quedarnos solo con transacciones donde el precio de la venta es mayor a 0. Nos enfocaremos en las compras reales, descartando otro tipo de transacciones."""

df_cleaned=df_grouped[df_grouped['price'] > 0]
df_cleaned[(df_cleaned['quantity'] < 0) & (df_cleaned['price'] <= 0)]

# Chequeo de cantidad de imputaciones
df_cleaned[df_cleaned['customer_id']==99999]

"""Vemos que posterior a remover filas, filas removidas que no representan ventas válidas, seguimos conservando la mayoría de los datos"""

df_cleaned[['price', 'quantity']]

df_cleaned[['price', 'quantity']].sort_values(by='quantity')

df_cleaned[['price', 'quantity']].describe()

cantidad_filas=df_grouped.shape[0]
cantidad_filas_limpieza=df_cleaned.shape[0]
print(f"Cantidad de filas antes de la limpieza: {cantidad_filas}")
print(f"Cantidad de filas despues de la limpieza: {cantidad_filas_limpieza}")
print(f"Porcentaje de filas removidas: {round((1-cantidad_filas_limpieza/cantidad_filas)*100, 2)}%")

"""### Creación de nuevas columnas

Comenzamos a agregar columnas con datos generados de los valores actuales para posterior análisis. Comenzamos con calcular el Monto total por item por compra.
"""

df_cleaned["monto_total"] = df_cleaned["quantity"] * df_cleaned["price"]
df_cleaned.head()

df_cleaned = agregar_momentos_temporales(df_cleaned)
df_cleaned.head()

df_cleaned[['dia_del_mes', 'hora']].describe()

print(df_cleaned.dia_de_semana.value_counts(),
      df_cleaned.momento_del_dia.value_counts(),
      df_cleaned.momento_del_mes.value_counts(),
      df_cleaned.momento_del_año.value_counts(),
      df_cleaned.mes.value_counts().sort_index()
      )

"""- No se encuentran registros los días Sábados, lo cual es extraño, analizaremos más adelante este tema.
- Cabe destacar que 2010Q4 es considerablemente menor al de 2011Q4 porque la fecha de inicio del dataset es 01-12-2010 y la fecha de fin 08-12-2011.
"""

df_cleaned.info()

"""En un principio pensamos en asignar los días festivos para cada uno de los países pero chequeando el de United Kindom confirmamos que no hay transacciones los días feriados."""

uk_holidays = holidays.UK()

# Agregar columna indicando si es festivo
df_grouped['es_festivo'] = False
df_grouped.loc[df_grouped.country =='United Kingdom',
               'es_festivo'] = df_grouped.loc[df_grouped.country=='United Kingdom',
                                              'date'].apply(lambda x: x in uk_holidays)
df_grouped['es_festivo'].value_counts()

"""## Categorización de productos

Dado que nuestro analisis quiere poder realizar una comparación por categorías, comenzamos a analizar la posibilidad de categorizar de forma manual.
"""

df_sales = df_cleaned.loc[df_cleaned.country=='United Kingdom',
 ['bill_no', 'item_name', 'date', 'price', 'customer_id',
  'quantity', 'monto_total', 'dia_del_mes', 'dia_de_semana',
  'momento_del_dia', 'hora', 'mes', 'momento_del_mes']]

df_sales.describe()

# Obtengo las cantidades segun Item
item = df_sales['item_name'].value_counts().reset_index()
item.columns = ['Item', 'Cantidad']
item

cat_features = ['item_name']
# Estadísticas de los posibles datos categóricos
df_sales[cat_features].describe()

"""Para mejorar el análisis del dataset de ventas, se aplicó una categorización sobre la columna item_name, con el objetivo de agrupar productos similares y reducir la dispersión causada por miles de ítems únicos.

Siguiendo un proceso de prueba y error, inicialmente más del 80% de los registros se clasificaban como “Otros”, dificultando la obtención de insights. Mediante un análisis de patrones léxicos y palabras clave, se definieron más de 25 categorías como “Decoración”, “Papelería y Regalos”, “Jardinería”, “Velas y Aromas”, entre otras.

Como resultado, se redujo la categoría “Otros” a menos del 5% del total de registros, lo que permite una segmentación más efectiva, una visualización clara de tendencias y una mejor toma de decisiones comerciales.

A continuación se ejecuta la categorización final.
"""

# Aplicamos la categorización segun la categoria, usamos a su vez el swifter dado que son muchos elementos, y esta libreria accelera el .apply
df_sales['categoria'] = df_sales['item_name'].swifter.apply(categorizar_item)
df_sales.head()

"""Parte de la logica utilizada para detectar las palabras más utilizadas, para poder crear categorías a partir de ello."""

# Unimos todos los textos en una sola cadena
todo_junto = ' '.join(item['Item'].dropna().astype(str).tolist()).lower()

# Dividimos por espacios
palabras = todo_junto.split()

# Contamos las palabras
conteo = Counter(palabras)

# Lo pasamos a DataFrame para ordenarlo
palabras_df = pd.DataFrame(conteo.items(), columns=['Palabra', 'Cantidad']).sort_values(by='Cantidad', ascending=False)

print(palabras_df.head(10))  # Las 10 más comunes

# Redefinimos item_name a minusculas
df_sales['item_name'] = df_sales['item_name'].str.lower()
df_sales['item_name']

cat_features = ['item_name']
# Estadísticas de los posibles datos categóricos
df_sales[cat_features].describe()

# Realizo count por categoria
df_sales['categoria'].value_counts()

# Dentro de los productos con categoría otros, vemos qué item_name hay, ordenados por cantidad
df_sales[df_sales['categoria'] == 'Otros'].item_name.value_counts()

"""Anexo: Posterior a las visualizaciones y análisis pudimos detectar que dada la concentración en los articulos de papeleria y regalo, nos preguntamos si efectivamente el negocio vendería articulos perecederos. Volvimos a investigar las categorías"""

lacteos = df_sales[df_sales['categoria'] == 'Lácteos']
lacteos.item_name.unique()

panificados = df_sales[df_sales['categoria'] == 'Panificados']
panificados.item_name.unique()

carnes = df_sales[df_sales['categoria'] == 'Carnes']
carnes.item_name.unique()

frutas = df_sales[df_sales['categoria'] == 'Frutas']
frutas.item_name.unique()

"""Nuestra conclusión y recomendación es poder tener un método de clásificación acorde para el caso de uso

## Analisis cantidad de filas eliminadas
"""

filas_inicial = len(df)
filas_final = len(df_grouped)
filas_final_uk = len(df_sales)
diferencia = filas_inicial - filas_final
diferencia_uk = filas_final - filas_final_uk
porcentaje_perdido = (diferencia / filas_inicial) * 100
porcentaje_perdido_uk = (diferencia_uk / filas_final) * 100

print(f"Filas iniciales: {filas_inicial}")
print(f"Filas finales post pre procesamiento: {filas_final}")
print(f"Filas finales solo UK: {filas_final_uk}\n")
print(f"Filas eliminadas (suma duplicados, eliminacion de no ventas): {diferencia}")
print(f"Filas eliminadas no UK: {diferencia_uk}\n")
print(f"Porcentaje filas eliminadas - pre procesamiento: {porcentaje_perdido:.2f}%")
print(f"Porcentaje filas eliminadas al enfocarnos en UK: {porcentaje_perdido_uk:.2f}%")

"""## Análisis calidad de los datos

El dataset incluye variables fundamentales para el análisis propuesto. Las variables (columnas) existentes permiten estudiar patrones de compra, comportamiento por cliente, estacionalidad, y asociaciones entre productos. Por lo tanto, los datos disponibles son pertinentes y adecuados para obtener resultados concluyentes respecto al comportamiento de compra en un contexto minorista.

**Completitud: ¿Faltan datos?**

Luego de un análisis inicial, se detecto que faltan datos a nivel de CustomerID, con la presencia de 26% de valores nulos. Imputamos, decidimos conservar estas transacciones anónimas para no perder casi un cuarto de los datos, imputando un ID genérico dado que el resto de campos son informativos.

Tambien se detectó valores nulos en Itemname, dado su % mínimo con respecto al dataset inicial, y no representaban ventas válidas los eliminamos.

Posterior pre procesamiento, mantuvimos un 98% de las filas, lo que representa una cantidad suficiente para realizar el análisis.

Por otro lado, la variable Country presenta una fuerte concentración de registros en un solo país: United Kingdom, que representa el 93% del total de transacciones en el data set original.

Para realizar un análisis comparativo entre paises, nos faltan datos.

Sin embargo, para realizar un análisis de tendencias de compra para United Kingdom no representa un inconveniente, dado la cantidad de datos que se conservan posterior al pre procesamiento. Siendo el porcentaje de filas eliminadas del 1.94%.

**Validez: ¿Cumplen con los formatos esperados?**

Sí. Luego del preprocesamiento, los datos fueron convertidos a sus tipos esperados.

Las fechas (Date) están en formato datetime

Los identificadores (CustomerID, BillNo) son enteros (int64)

El campo Itemname fue transformados a tipo string.

Country podría ser categorico, pero dada la consentración en 'United Kingdom', decidimos no transformarlo.

Se eliminaron o corrigieron valores con caracteres no válidos (letras en campos numéricos, símbolos, etc.).

Se transformo a float la columna Price, luego de reamplazar "," por ".".

Se imputaron los valores nulos de CustomerID a 99999.

**Precisión: ¿Son correctos y fiables?**

Se asumió que los datos provienen de una fuente confiable. No obstante, se identificaron y excluyeron posibles errores como:

Cantidades y precios negativos. Item name nulo, donde CustomerID era nulo y precio 0, identificamos que no correspondian a transacciones válidas.

Registros con nombres de ítems que evidentemente no corresponden a una venta (como "Adjust bad debt").

**Consistencia: ¿Coinciden los datos entre fuentes?**

En nuestro caso no aplica, dado que estamos utilizando una sola fuente.

**Relevancia**: ¿Los datos son pertinentes para tener resultados
concluyentes del problema en estudio?

El dataset contiene información de transacciones de un minorista, incluyendo:

*   Productos (Itemname)
*   Fechas (Date)
*   Clientes (CustomerID)
*   Precio (Price)
*   País (Country)
*   Cantidad (Quantity)


Estas variables son parte fundamental del problema que estamos estudiando.

**Unicidad: ¿Ausencia de registros duplicados?**

Detectamos filas duplicadas dentro de una misma compra, y segun el análisis de la distribución, concluímos no eliminarlos, y sumarlos a la cantidad total de productos para ese item en su compra correspondiente. Ademas, de cara a un análisis de canasta (relaciones entre productos), habría que considerar cada producto único por factura


**Integridad: ¿Datos completos y sin corrupción?**

Se verificó que los campos clave (BillNo, Itemname, Quantity, Price, Date) no estén vacíos ni corruptos.

También se validó:

Que las fechas sean válidas y no futuras

Que los valores de cantidad y precio sean positivos

Que no haya ítems malformateados

Se removieron registros administrativos o contables que no representan una compra real

## Resultados y discusión

Se evaluó la calidad del dataset considerando criterios clave. Se imputaron los valores faltantes en CustomerID por su relevancia y se eliminaron registros inválidos con Itemname nulo. Tras el preprocesamiento, se conservó el 98% de las filas. La reducción de filas no se debió únicamente a la eliminación de datos, sino también a la consolidación de cantidades de un mismo ítem dentro de la misma factura, una transformación necesaria para realizar un análisis de canasta de mercado (basket analysis), donde se analiza la combinación de productos adquiridos por transacción.

Se aseguraron los formatos esperados en todas las columnas, corrigiendo errores como símbolos en campos numéricos y formateo incorrecto de precios. Se mantuvieron duplicados válidos dentro de una misma compra, y se descartaron registros administrativos no representativos. Los datos finales presentan integridad, validez y consistencia suficientes para realizar un análisis confiable. Además, se agregó una categorización manual de los productos para permitir una futura visualización por grupos. Dada la gran cantidad de filas, esta agrupación resulta necesaria para facilitar un análisis de tendencias más claro y enfocado. Por otro lado, y con el objetivo de detectar estacionalidades en los patrones de compra, se generaron también nuevas variables categóricas que identifican el momento del día, el momento del mes y el mes en que se realizó cada transacción.

Como existe una concentración del 93% en United Kingdom, se decidió focalizar el análisis en este país. Esta decisión se basa en que, luego del preprocesamiento, se conservaron 510,491 filas de las 520,606 originales, y al filtrar exclusivamente por registros del Reino Unido, se conservan 475,171 observaciones. Esto representa un volumen suficientemente robusto como para permitir un análisis detallado de tendencias y la generación de recomendaciones específicas para este mercado.

# Etapa III: Visualización y análisis de datos

## **Visualización**

### Distribución de los precios y cantidad de ventas

Para facilitar la comprensión de los productos más vendidos y aquellos que generan mayores ingresos, la visualización individual por ítem resulta compleja. Por lo tanto, recurrimos a las categorías previamente creadas para una mejor interpretación de los datos.

#### Precios

Antes que nada queremos observar la distribución de los precios por categoría para detectar la presencia de outliers y si las distintas categorías se comportan de manera similar.
"""

sns.boxplot(data=df_sales, x="price", y="categoria")

"""Como en la categorías 'Papelería y Regalos', 'Ornamentos y Figuras' y 'Otros' se detecta la presencia de valores muy altos, se hace difícil la visualización de los boxplot, por lo tanto decidimos excluirlas del gráfico."""

sns.boxplot(data=df_sales.loc[((df_sales.categoria!='Papelería y Regalos') &
                               (df_sales.categoria!='Ornamentos y Figuras') &
                               (df_sales.categoria!='Otros'))],
            x="price",
            y="categoria")

"""Ahora podemos observar con mayor claridad las diferencias entre las categorías, por ejemplo **Panificados** tiene una media superior al resto. Si queremos ver aún más en detalle podemos filtrar por precio como se ve en la siguiente gráfica. Algunas categorías se comportan de manera similar, tienen medias y cuantiles parecidos y otras son bien distintas como **Muebles** que por ejemplo tiene un mínimo mucho más alto que el resto."""

sns.boxplot(data=df_sales.loc[df_sales.price<50], x="price", y="categoria")

"""#### Cantidad de ventas

Ahora vamos a hacer lo mismo pero con la cantidad de ventas, como sucede algo similar, en el gráfico siguiente vamos a excluir 'Ornamentos y Figuras' y 'Otros' para visualizar mejor las distribuciones.
"""

sns.boxplot(data=df_sales, x="quantity", y="categoria")

sns.boxplot(data=df_sales.loc[((df_sales.categoria!='Ornamentos y Figuras') &
                               (df_sales.categoria!='Otros'))],
            x="quantity",
            y="categoria")

"""Como sigue siendo difícil ver las cajas de las medias y cuantiles porque la distribuciones tienen una cola hacia la derecha muy extensa, filtramos también por cantidad. Esto nos permite ver que en relación a la media, el quantil 25 y el 75 hay categorías con comportamientos similares. Muebles parece ser la que tiene valores más bajos con respecto a estas métricas."""

sns.boxplot(data=df_sales.loc[df_sales.quantity<50], x="quantity", y="categoria")

"""De esta exploración comenzamos a ver que existen outliers por categoría a nivel de precios y cantidad. Se observa una gran dispersión de los datos, por lo que seguiremos en otro tipo de visualizaciones y posterior analisis de outliers.

### Ventas por Categoría

#### Cantidad de Ventas por Categoría
"""

ventas_categorias = (df_sales.groupby('categoria').
                     agg({'quantity': 'sum'}).
                     reset_index().
                     sort_values(by='quantity', ascending=False))
ventas_categorias.head(10)

ventas_categorias_top = ventas_categorias.iloc[:20,:]
ventas_categorias_top.head(6)

"""En el siguiente gráfico podemos observar que 'Papelería y Regalos', 'Otros' y 'Bolsas y Organizadores' son las que ocupan los primeros lugares con más 500.000 unidades vendidas. La categoría 'Otros' si bien se encuentra en el segundo lugar, agrupa muchos productos por lo tanto es de esperar que esté en los primeros lugares. Los últimos 10 lugares muestran una distribución mucho más uniforme que los primeros 10."""

graficar_barras_horizontales_con_valores(
    data=ventas_categorias_top,
    columna_x="quantity",
    columna_y="categoria",
    titulo="Cantidad de Ventas por Top 20 de Categoría",
    xlabel="Cantidad de Ventas",
    ylabel="Categoría"
)

"""En este gráfico podemos observar que los primeros cuatro productos tienen una cantidad de ventas muy alta con respecto al resto.

#### Cantidad de Ventas por Item
"""

top_cat = ["Papelería y Regalos", "Otros", "Bolsas y Organizadores", "Decoración", "Ornamentos y Figuras", "Decoración de Pared"]

# Agrupar por producto y sumar la cantidad vendida
top_items = df_sales.groupby("item_name")["quantity"].sum().sort_values(ascending=False).reset_index().head(20)
top_items

graficar_barras_horizontales_con_valores(
    data=top_items,
    columna_x="quantity",
    columna_y="item_name",
    titulo="Cantidad de Ventas por Top 20 de Item",
    xlabel="Cantidad de Ventas",
    ylabel="Item"
)

top_items = top_items.iloc[0:11]
top_items

"""Para mejor visualización utilizaremos el top 10"""

graficar_barras_horizontales_con_valores(
    data=top_items,
    columna_x="quantity",
    columna_y="item_name",
    titulo="Cantidad de Ventas por Top 10 de Item",
    xlabel="Cantidad de Ventas",
    ylabel="Item"
)

"""En este gráfico podemos observar que los primeros cuatro productos tienen una cantidad de ventas muy alta con respecto al resto.

Posterior a estos análisis, vemos una concentración en las ventas de ciertas categorías e items. Ahora queremos identificar si esto sucede en cierto período, justificado por un evento o estación.
"""

# Agrupar por producto y sumar la cantidad vendida
dias_ventas = df_sales.groupby("dia_de_semana")["quantity"].sum().sort_values(ascending=False).reset_index()
orden_categorias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dias_ventas.loc[len(dias_ventas)] = ["Saturday", 0]

# Ordenar por dia
categoria_ordenada = pd.CategoricalDtype(categories=orden_categorias, ordered=True)
dias_ventas["dia_de_semana"] = dias_ventas["dia_de_semana"].astype(categoria_ordenada)
dias_ventas = dias_ventas.sort_values(by="dia_de_semana").reset_index(drop=True)

dias_ventas

"""En esta búsqueda volvemos a reafirmar que los sábados no hay ingresos. El motivo de esto no es desconocido dado que no sabemos el contexto del negocio."""

graficar_barras_horizontales_con_valores(
    data=dias_ventas,
    columna_x="quantity",
    columna_y="dia_de_semana",
    titulo="Cantidad de Ventas por dia de semana",
    xlabel="Cantidad de Ventas",
    ylabel="Item"
)

"""Asi como hemos observado la distribución entre días, vemos la necesidad de investigar lo mismo para las horas. En que rango de horas hay mayor concentración de compras?"""

## Calcular el total de transacciones por hora
ordenes_por_hora = df_sales.groupby('hora')['bill_no'].nunique()

plt.figure(figsize=(12, 6))
plt.plot(ordenes_por_hora.index, ordenes_por_hora.values, marker='o')
plt.title('Total de Ventas por Hora del Día')
plt.xlabel('Hora')
plt.ylabel('Cantidad de Órdenes')
plt.xticks(range(5, 21))
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Evolución mensual

Tras analizar la distribución de las variables mediante boxplots y visualizar el top de categorías y productos, se incorpora la dimensión temporal con el objetivo de observar la evolución de estos indicadores a lo largo del tiempo.

Dado que el conjunto de datos solo incluye los primeros 9 días de diciembre de 2011, se optó por excluir este mes de las visualizaciones para asegurar una comparación adecuada y homogénea entre períodos. Por lo tanto, el análisis temporal se centrará en el intervalo comprendido entre diciembre de 2010 y noviembre de 2011, meses para los cuales se dispone de información completa.
"""

df_sales.date.min(), df_sales.date.max()

df_sales_date = df_sales.loc[df_sales.date<'2011-12-01']
df_sales_date.date.max()

"""#### Ingresos por Ventas"""

ventas_mensuales = df_sales_date.groupby(df_sales_date['mes']).agg({'monto_total': 'sum'})
ventas_mensuales.describe()

graficar_xy(
    ventas_mensuales,
    titulo='Evolución Mensual de las Ventas',
    label_y='Monto Total',
    formato_y_en_miles=True
    )

"""En este gráfico podemos observar que el monto total de ventas desde diciembre de 2010 hasta agosto de 2011 es bastante estable pero a partir de setiembre empieza a subir considerablemente. En los siguientes gráficos vamos a investigar para tratar de encontrar la razón de esta suba.

#### Cantidad de transacciones

La primera dimensión que exploraremos, será la cantidad de transacciones, con el objetivo de determinar si el aumento en los ingresos totales del comercio se relaciona con la cantidad de ventas realizadas, o si se debe a un aumento del ticket promedio (el cual a su vez podría ser por un aumento de precios o de las cantidades compradas dentro de cada transacción)
"""

df_transacciones = df_sales_date[['mes', 'bill_no']].drop_duplicates()
transacciones_mensuales = df_transacciones.groupby(df_transacciones['mes']).agg({'bill_no': 'count'})
transacciones_mensuales.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)
transacciones_mensuales

# Gráfico de evolución mensual
graficar_xy(
    data=transacciones_mensuales,
    columna_x='mes',
    columna_y='cantidad_transacciones',
    titulo='Evolución Mensual de Cantidad de Transacciones',
    color='red',
    label_y='Cantidad de Transacciones',
)

graficar_doble_eje_y(
    data1=transacciones_mensuales,
    x1='mes',
    y1='cantidad_transacciones',
    label_y1='Cantidad de Transacciones',
    color1='red',

    data2=ventas_mensuales,
    y2='monto_total',
    label_y2='Ventas Totales (en miles)',
    color2='green',

    titulo='Evolución Mensual: Ventas Totales y Transacciones',
    formato_y2_en_miles=True
)

"""Como la cantidad de transacciones sigue la misma tendencia que el monto total de ventas (y sabemos que este es precio x cantidad), podemos deducir que la suba en el monto total se explica casi en su totalidad por la cantida de transacciones que por variaciones el precio.

Por otro lado, el mes en el que más divergen ambas gráficas es enero de 2011, como vemos que la cantidad de transacciones baja más que el monto total, se desprende que en ese mes el ticket promedio aumentó.

#### Cantidad distinta de productos por transacción

Nos proponemos investigar ahora si la suba puede deberse a que haya más cantidad de productos disponibles, esto no podemos saberlo dado que no manejamos el dataset del stock del local. Pero si podemos saber si se compraron más cantidad de items distintos por transacción.

¿Cuántos productos distintos compra un cliente en promedio por transacción en cada mes?
"""

df_productos = df_sales_date[['mes', 'bill_no', 'item_name']].drop_duplicates()

productos_mensuales = df_productos.groupby(['mes','bill_no']).agg({'item_name': 'count'})
productos_mensuales.rename(columns={'item_name': 'cantidad_items_distintos'}, inplace=True)

productos_mensuales_avg = productos_mensuales.groupby('mes').agg({'cantidad_items_distintos': 'mean'}).round(2)
productos_mensuales_avg

# Gráfico de evolución mensual
graficar_xy(
    data=productos_mensuales_avg,
    columna_x='mes',
    columna_y='cantidad_items_distintos',
    titulo='Evolución Mensual de Cantidad de Items distintos vendidos',
    color='blue',
    label_y='Promedio de items distintos por Bill_no',
)

graficar_doble_eje_y(
    data1=transacciones_mensuales,
    x1='mes',
    y1='cantidad_transacciones',
    label_y1='Cantidad de Transacciones',
    color1='red',

    data2=productos_mensuales_avg,
    y2='cantidad_items_distintos',
    label_y2='Promedio de items distintos por Bill_no',
    color2='blue',

    titulo='Evolución Mensual: Promedio Items Distintos y Transacciones'
)

"""Con respecto al aumento de transacciones en los últimos meses, si bien ambas variables —la cantidad de transacciones y el promedio de ítems distintos por transacción— parecen moverse en la misma dirección, no se observa una correlación clara entre ellas, ya que su comportamiento en los meses anteriores es bastante dispar.

Llama la atención que en enero de 2011 se registra una leve disminución en la cantidad de transacciones, pero un incremento en el promedio de ítems distintos por transacción. Esto podría deberse a la presencia de algún valor atípico, como una compra excepcionalmente grande.

Otro mes destacable es mayo de 2011, donde ocurre lo opuesto: aumentan las transacciones, pero el promedio de ítems distintos por transacción alcanza su punto más bajo, lo cual sugiere un mayor número de compras que incluyen los mismos productos repetidos.

#### Cantidad de Clientes

Siguiendo con la misma linea de investigación, queremos determinar si esta tendencia se debe a que el negocio aumentó su cantida de clientes o si es que estos realizan mayor cantidad de compras.
"""

df_clientes = df_sales_date[['mes', 'customer_id']].drop_duplicates()
clientes_mensuales = df_clientes.groupby(['mes']).agg({'customer_id': 'count'})
clientes_mensuales.rename(columns={'customer_id': 'cantidad_customer_id'}, inplace=True)

# Gráfico de evolución mensual
graficar_xy(
    data=clientes_mensuales,
    columna_x='mes',
    columna_y='cantidad_customer_id',
    titulo='Evolución Mensual de Cantidad de Clientes Distintos',
    color='violet',
    label_y='Cantidad de Clientes'
)

"""Dado que la evolución de la cantidad de clientes muestra la misma tendencia, presumimos que esta es la explicación del aumento, vamos a compararlo con la siguiente gráfica."""

graficar_doble_eje_y(
    data1=transacciones_mensuales,
    x1='mes',
    y1='cantidad_transacciones',
    label_y1='Cantidad de Transacciones',
    color1='red',

    data2=clientes_mensuales,
    y2='cantidad_customer_id',
    label_y2='Cantidad de Clientes',
    color2='violet',

    titulo='Evolución Mensual: Cantidad de Clientes y Transacciones'
)

graficar_doble_eje_y(
    data1=clientes_mensuales,
    x1='mes',
    y1='cantidad_customer_id',
    label_y1='Cantidad de Clientes',
    color1='violet',

    data2=ventas_mensuales,
    y2='monto_total',
    label_y2='Ventas Totales (en miles)',
    color2='green',

    titulo='Evolución Mensual: Ventas Totales y Cantidad de Clientes',
    formato_y2_en_miles=True
)

"""Efectivamente al superponer ambas gráficas podemos observar que, si bien en algunos meses (como diciembre/10, marzo/11 o agosto-octubre/11), ambas líneas se distancian levemente, la tendencia general que siguen es la misma.
Esto significa que las ventas del comercio se relacionan directamente con la cantidad de clientes que compran en él.


Por otro lado, en cuanto a la pregunta de si el aumento de las ventas totales se relacionan a un aumento de clientes, o un aumento de compras por cliente, a partir de este resultado podemos presumir que la cantidad de transacciones promedio por cliente no guarda relación con el monto total vendido, y, que de hecho, como ambas líneas están casi superpuestas, podemos concluir que la cantidad promedio de compras por cliente es relativamente estable a lo largo de todo el período. Revisaremos esto último en el siguiente apartado.

#### Cantidad de compras por cliente

Para analizar la cantidad de compras por cliente decidimos excluir el '99999' ya que fue el valor que imputamos en aquellas compras que no tenían numero de cliente.
"""

df_compras_clientes = df_sales_date[['mes', 'customer_id', 'bill_no']].drop_duplicates()

compras_clientes_mensuales = df_compras_clientes.groupby(['mes', 'customer_id']).agg({'bill_no': 'count'}).reset_index()
compras_clientes_mensuales.rename(columns={'bill_no': 'cantidad_compras_cliente'}, inplace=True)

cantidad_99999 = df_compras_clientes[df_compras_clientes['customer_id'] == 99999].shape[0]
print(f"Hay {cantidad_99999} filas con customer_id = 99999.")
total = len(df_compras_clientes)
porcentaje_99999 = round((cantidad_99999 / total) * 100, 2)
print(f"Eso representa el {porcentaje_99999}% del total de filas ({total}).")

compras_clientes_mensuales = compras_clientes_mensuales.loc[compras_clientes_mensuales.customer_id != 99999]
compras_clientes_mensuales

compras_clientes_mensuales_avg = compras_clientes_mensuales.groupby('mes').agg({'cantidad_compras_cliente': 'mean'}).round(2)
compras_clientes_mensuales_avg

# Gráfico de evolución mensual
graficar_xy(
    data=compras_clientes_mensuales_avg,
    columna_x='mes',
    columna_y='cantidad_compras_cliente',
    titulo='Evolución Mensual de Cantidad de Compras Promedio por Cliente',
    color='blue',
    label_y='Promedio de Compras por Cliente',
    limites_y=(1, 2)
)

"""Dado que la variación del promedio de compras por clientes es muy baja (se mantiene entre uno y dos todos los meses), confirmamos que es por por la suba de clientes. Los puntos más altos se explican por algún cliente que en estos meses particulares tienen muchas compras.

Investigaremos esto último.
"""

compras_clientes_mensuales_max = compras_clientes_mensuales.groupby('mes').agg({'cantidad_compras_cliente': 'max'})
compras_clientes_mensuales_max

"""Investigamos el top 3 de clientes compradores por mes para corroborar si un mismo cliente registra compras altas todos los meses (o si varía período a período)"""

# Agrupar por mes y cliente, sumando la cantidad de compras
compras_por_cliente_mes = compras_clientes_mensuales.groupby(['mes', 'customer_id']) \
                                                    .agg({'cantidad_compras_cliente': 'sum'}) \
                                                    .reset_index()

# Ordenar por mes y cantidad de compras descendente
compras_ordenadas = compras_por_cliente_mes.sort_values(['mes', 'cantidad_compras_cliente'], ascending=[True, False])
compras_ordenadas

# Crear columna combinada de mes + cliente para que cada barra sea única
top3_clientes_por_mes = compras_ordenadas.groupby('mes').head(3).copy()

top3_clientes_por_mes["mes_cliente"] = (
    top3_clientes_por_mes["mes"] + " - " + top3_clientes_por_mes["customer_id"].astype(str)
)
top3_clientes_por_mes.sort_values('customer_id').head()

plt.figure(figsize=(14, 6))
sns.barplot(
    data=top3_clientes_por_mes,
    x="mes_cliente",
    y="cantidad_compras_cliente",
    palette="viridis"
)
plt.xticks(rotation=45, ha="right")
plt.title("Top 3 Clientes por Mes - Compras")
plt.ylabel("Cantidad de Compras")
plt.xlabel("Mes - Cliente")
plt.tight_layout()
plt.show()

"""Efectivamente vemos que varios de los clientes aparecen en más de un mes dentro del top 3 de compradores, donde los clientes 12749 y 17841 se destacan por aparecer en el top en 9 y 8 de los 12 meses analizados.

A su vez, se observan picos altos en los meses donde vimos anteriormente picos en los promedios de compras. Esto demuestra nuestra primera teoría de que esos picos no se debían a un aumento general de la cantidad de compras por cliente, sino que estaba influenciado por unos pocos clientes.

Para visualizar lo anterior de manera más clara, graficamos en una línea temporal la cantidad máxima correspondientes a un mismo cliente, por mes. Si los picos que veíamos fueran ocasionados por un aumento general en el promedio de compras por clientes, las cantidades máximas no deberían guardar relacion con la gráfica de más arriba. Por el contrario, si los picos se dan en las mismas fechas en que los máximos tambien aumentan, significa que son éstos los que están influenciando la gráfica general.
"""

# Gráfico de evolución mensual
graficar_xy(
    data=compras_clientes_mensuales_max,
    columna_x='mes',
    columna_y='cantidad_compras_cliente',
    titulo='Evolución Mensual de Máxima Cantidad de Compras por Cliente',
    color='blue',
    label_y='Máxima Cantidad de Compras por Cliente',
    formato_y_en_miles=False,
    limites_y=(0, 50)
)

"""Finalmente podemos determinar que, la cantidad de compras promedio por cliente se mantuvo estable durante todo el período, y que, los meses en los que ésta aumentó, se deben a unos pocos clientes que realizaron una cantidad de compras sensiblemente superior."""

graficar_doble_eje_y(
    data1=transacciones_mensuales,
    x1='mes',
    y1='cantidad_transacciones',
    label_y1='Cantidad de Transacciones',
    color1='red',

    data2=compras_clientes_mensuales_max,
    y2='cantidad_compras_cliente',
    label_y2='Máxima Cantidad de Compras por Cliente',
    color2='blue',

    titulo='Evolución Mensual: Max Compras Clientes y Transacciones'
)

"""Si bien se puede ver una similitud entre ambas gráficas, consideramos que podría tratarse de una relación espúrea, ya que no hay fundamento teórico para justificar que estas dos variables se expliquen una a la otra (teniendo en cuenta que los promedios de compras por cliente se mantienen estables).

#### Cantidad de transacciones por el top de categorías

En este apartado nos proponemos investigar si hay alguna categoría de productos que contribuya en mayor medida a la tendencia que muestran las ventas totales, con el fin de identificar los grupos de productos más pujantes del inventario.
"""

df_transacciones_cat = df_sales_date[['mes', 'categoria', 'bill_no']].drop_duplicates()
transacciones_mensuales_cat = df_transacciones_cat.groupby(['mes', 'categoria']).agg({'bill_no': 'count'}).reset_index()
transacciones_mensuales_cat.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)
transacciones_mensuales_cat

transacciones_mensuales_cat_top = transacciones_mensuales_cat.loc[transacciones_mensuales_cat.categoria.isin(top_cat)]
transacciones_mensuales_cat_top

# Gráfico de evolución mensual
graficar_xy(
    data=transacciones_mensuales_cat_top,
    columna_x='mes',
    columna_y='cantidad_transacciones',
    titulo='Evolución Mensual de Cantidad de Transacciones por Top de Categoría',
    label_y='Cantidad de Transacciones',
    color='blue',
    hue='categoria'
)

"""Si estudiamos la evolución del total de transacciones por el top de categorías vemos un comportamiento similar entre ellas y similar a la tendencia general.  Si bien la categoría "Papelería y Regalos" muestra un nivel de ventas superior al resto durante todo el período, no vemos, a nivel de cantidad de transacciones, una que por su peso y comportamiento, se muestre más influyente que el resto sobre la tendencia general.

#### Monto total por el top de categorías

Comprobaremos ahora, si desde la dimensión del monto podemos encontrar una categoría más influyente que las demás sobre los ingresos totales por ventas.
"""

df_monto_cat = df_sales_date[['mes', 'categoria', 'monto_total']]
monto_cat_mensuales = df_monto_cat.groupby(['mes', 'categoria']
                                           ).agg({'monto_total': 'sum'}).reset_index()
monto_cat_mensuales.rename(columns={'monto_total': 'monto_total_categoria'}, inplace=True)
monto_cat_mensuales

top_cat = ventas_categorias.head(6).categoria.unique()
monto_cat_mensuales_top = monto_cat_mensuales.loc[monto_cat_mensuales.categoria.isin(top_cat)]
monto_cat_mensuales_top

# Gráfico de evolución mensual
graficar_xy(
    data=monto_cat_mensuales_top,
    columna_x='mes',
    columna_y='monto_total_categoria',
    titulo='Evolución Mensual de Monto Total de Ventas por Top de Categoría',
    label_y='Monto Total de Ventas',
    formato_y_en_miles=True,
    hue='categoria'
)

"""Al analizar el gráfico de evolución mensual del monto total de ventas por categoría, observamos que la categoría 'Papelería y Regalos' muestra un crecimiento sostenido en los últimos meses, destacándose como la principal responsable del aumento general del monto total. Este comportamiento no se replica en el resto de las categorías, que presentan una tendencia más estable e incluso estancada.

Este contraste llama la atención especialmente porque, a pesar de que la cantidad de transacciones aumentó en ese período, el monto total para las demás categorías no se incrementó en la misma proporción. Esto sugiere que podrían haberse producido cambios en el comportamiento de compra: una posible reducción en los precios o una disminución en la cantidad de ítems comprados por transacción en esas categorías.

Este tipo de análisis permite enfocar futuras estrategias comerciales en las categorías con mayor impacto estacional o de crecimiento, como 'Papelería y Regalos', y revisar el desempeño de las más estables.

#### Cantidad de transacciones por semana

La evolución semanal de las ventas nos permite observar con mayor detalle cuáles son las semanas en las que hay más ventas. Pasaremos a hacer un análisis semanal.
"""

df_sales_date['date'] = pd.to_datetime(df_sales_date['date'])

# Calcular el lunes de la semana correspondiente
df_sales_date['week'] = df_sales_date['date'] - pd.to_timedelta(df_sales_date['date'].dt.weekday, unit='d')

# Formatear al formato deseado '%Y-%m-%d'
df_sales_date['week'] = df_sales_date['week'].dt.strftime('%Y-%m-%d')
df_sales_date.head()

df_sales_date['week'] = pd.to_datetime(df_sales_date['week']).dt.strftime('%Y-%m-%d')
df_transacciones_semana= df_sales_date[['week', 'bill_no']].drop_duplicates()
transacciones_semanales = df_transacciones_semana.groupby(['week']).agg({'bill_no': 'count'}).reset_index()
transacciones_semanales.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)
transacciones_semanales.head()

# Gráfico de evolución mensual
graficar_xy(
    data=transacciones_semanales,
    columna_x='week',
    columna_y='cantidad_transacciones',
    titulo='Evolución de Cantidad de Transacciones por Semana',
    label_y='Cantidad de Transacciones',
    color='orange'
)

"""Al observar la evolución semanal de la cantidad de transacciones, notamos un crecimiento general hacia los últimos meses del período analizado. Sin embargo, este aumento no es completamente sostenido ni uniforme. El gráfico muestra picos específicos en semanas concretas, lo que sugiere que los incrementos en las transacciones no siempre responden a una tendencia de crecimiento orgánico, sino que podrían estar influenciados por acciones puntuales, como promociones, campañas o eventos especiales.

Este comportamiento se evidencia claramente en semanas como la del 6 de diciembre de 2010 y la del 9 de mayo de 2011, donde se registran aumentos abruptos en la cantidad de transacciones, seguidos por caídas en las semanas posteriores. Estos picos atípicos refuerzan la hipótesis de que determinadas campañas o fechas festivas que podrían estar generando aumentos temporales en la actividad comercial.

En este contexto, el análisis semanal complementa al mensual, permitiendo detectar eventos puntuales que no serían visibles con una agregación más amplia.

#### Valor del ticket promedio

Para comprender mejor el comportamiento de compra de los clientes a lo largo del tiempo, se analiza el ticket promedio. De esta forma ir identificando tendencias estacionales, detectar anomalías en el comportamiento de los consumidores y evaluar la relación entre el gasto promedio por transacción y el volumen general de actividad comercial.
"""

# Calcular monto total por factura (por si no está correctamente consolidado)
# Esto suma los montos por cada bill_no (factura)
facturas = df_sales_date.groupby(['bill_no', 'dia_del_mes', 'week', 'mes'])['monto_total'].sum().reset_index(name='total_factura')

# Ticket promedio por día
ticket_diario = facturas.groupby(['dia_del_mes', 'mes'])['total_factura'].mean().reset_index(name='ticket_promedio_dia')

# Ticket promedio por semana
ticket_semanal = facturas.groupby('week')['total_factura'].mean().reset_index(name='ticket_promedio_semana')

# Ticket promedio por mes
ticket_mensual = facturas.groupby('mes')['total_factura'].mean().reset_index(name='ticket_promedio_mes')

graficar_xy(
    data=ticket_semanal,
    columna_x='week',
    columna_y='ticket_promedio_semana',
    titulo='Evolución del Ticket Promedio',
    label_y='Ticket Promedio Semana',
    color='red'
)

graficar_xy(
    data=ticket_mensual,
    columna_x='mes',
    columna_y='ticket_promedio_mes',
    titulo='Evolución del Ticket Promedio',
    label_y='Ticket Promedio Mes',
    color='red'
)

graficar_doble_eje_y(
    data1=ticket_semanal,
    x1='week',
    y1='ticket_promedio_semana',
    label_y1='Ticket Promedio',
    color1='magenta',

    data2=transacciones_semanales,
    y2='cantidad_transacciones',
    label_y2='Cantidad de Transacciones',
    color2='green',

    titulo='Evolución Semana: Ticket Promedio y Cantidad de Transacciones'
)

graficar_doble_eje_y(
    data1=ticket_mensual,
    x1='mes',
    y1='ticket_promedio_mes',
    label_y1='Ticket Promedio',
    color1='magenta',

    data2=transacciones_mensuales,
    y2='cantidad_transacciones',
    label_y2='Cantidad de Transaccions',
    color2='green',

    titulo='Evolución Mensual: Ticket Promedio y Cantidad de Transacciones'
)

graficar_doble_eje_y(
    data1=ticket_mensual,
    x1='mes',
    y1='ticket_promedio_mes',
    label_y1='Ticket Promedio',
    color1='magenta',

    data2=ventas_mensuales,
    y2='monto_total',
    label_y2='Monto Total',
    color2='green',
    formato_y2_en_miles=True,

    titulo='Evolución Mensual: Ticket Promedio y Ventas'
)

"""Cuando el ticket promedio aumenta mientras que tanto la cantidad de transacciones como el monto total de ventas disminuyen, esto puede interpretarse como un cambio en el comportamiento de compra de los clientes:

Menos clientes, compras más grandes: Una posible explicación es que hubo menos transacciones (es decir, menos clientes o menos tickets emitidos), pero los que sí compraron realizaron compras de mayor valor.

Caída en volumen, pero no en valor por operación: A nivel de negocio, esta situación podría ser positiva si el margen de ganancia es mayor en productos más caros. Sin embargo, la caída en el número de transacciones y en el monto total advierte sobre una posible disminución del flujo de clientes.

Esta señal sugiere que estrategias orientadas a aumentar el número de transacciones podrían ser necesarias si se desea mantener o incrementar el volumen total de ventas, sin depender únicamente del aumento en el valor de cada ticket.

### Mapa de calor
"""

# hour
df_sales["hora"] = pd.to_datetime(df_sales["date"])
df_sales["hora"] = df_sales["hora"].dt.hour
df_sales.head()

graficar_heatmap(
    data=df_sales,
    fila="dia_de_semana",
    columna="hora",
    valor="bill_no",
    aggfunc="nunique",
    figsize=(15,7),
    orden_filas=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    orden_columnas=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    titulo="Cantidad de de ventas",
    xlabel="Hora",
    ylabel="Día de la semana"
)

"""Con el objetivo de identificar patrones temporales en el comportamiento de compra, se construyó un mapa de calor que muestra el monto total de ventas segmentado por día de la semana y momento del día (mañana, tarde y noche). Esta visualización permite detectar de manera rápida los períodos con mayor actividad comercial, así como posibles oportunidades para optimizar promociones, horarios de atención o gestión de recursos."""

graficar_heatmap(
    data=df_sales,
    fila="dia_de_semana",
    columna="momento_del_dia",
    valor="monto_total",
    orden_filas=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    orden_columnas=["Mañana", "Tarde", "Noche"],
    titulo="Monto total de ventas",
    xlabel="Momento del día",
    ylabel="Día de la semana"
)

"""El gráfico refuerza una observación ya detectada durante el preprocesamiento: no se registran datos para los sábados. Esto podría deberse a que el comercio no opera ese día, o a que las transacciones correspondientes se cargan en el sistema en otros momentos, como los domingos. Estas hipótesis surgen de lo observado en los datos.

Ademas, el mapa de calor permite identificar que el período con mayor monto total de ventas corresponde a los martes en la tarde. No obstante, este hallazgo debe interpretarse con cautela, ya que no es posible determinar si esta tendencia se repite de forma consistente a lo largo del tiempo o si responde a picos puntuales en uno o más meses específicos.

En consecuencia, se continuará indagando este aspecto para lograr una interpretación fundamentada.
"""

graficar_heatmap(
    data=df_sales_date,
    fila="dia_de_semana",
    columna="mes",
    valor="monto_total",
    orden_filas=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    orden_columnas=['2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05',
                    '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11'],
    titulo="Monto total de ventas",
    xlabel="Mes",
    ylabel="Día de la semana"
)

"""Un fenómeno similar se observa al analizar los datos correspondientes a noviembre de 2011, donde los lunes se destacan como los días con mayor monto total de ventas. Sin embargo, para interpretar correctamente este comportamiento, es necesario profundizar el análisis dentro del mes. El objetivo es determinar si esta concentración de ventas se distribuye de manera uniforme entre todos los lunes de noviembre, o si responde a un único día con valores atípicamente altos que esté sesgando el promedio mensual."""

df_noviembre = df_sales_date[df_sales_date["mes"] == "2011-11"].copy()

graficar_heatmap(
    data=df_noviembre,
    fila="dia_de_semana",
    columna="week",
    valor="monto_total",
    aggfunc="sum",
    orden_filas=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    orden_columnas=sorted(df_noviembre["week"].unique()),
    titulo="Ventas semanales por día de la semana (Noviembre 2011)",
    xlabel="Semana",
    ylabel="Día de la semana",
    figsize=(10, 6),
    cmap="YlGnBu"
)

"""Se identificó que una semana específica de noviembre presenta un pico destacado en el volumen de ventas, concentrado exclusivamente en el día lunes. Este comportamiento sugiere la ocurrencia de un evento puntual, como una promoción, campaña comercial o acción estacional específica, que generó un flujo atípico de transacciones en esa fecha.

Este hallazgo resulta relevante desde una perspectiva operativa, ya que permite anticipar comportamientos similares en años futuros. Contar con esta información facilita la planificación de stock y recursos, optimizando la preparación del negocio ante posibles picos de demanda en fechas estratégicas.

### Grafico Mosaico

Para nuestro caso el siguiente gráfico de mosaico no aporta mayor información a lo que venimos trabajando y puede ser descartado en favor de visualizaciones más informativas.
"""

data_mosaico = df_sales.groupby(["dia_de_semana", "momento_del_dia"]).size()

# Convertir a formato requerido por mosaic
data_dict = {(dia, momento): valor for (dia, momento), valor in data_mosaico.items()}

# Crear el gráfico
plt.figure(figsize=(12, 6))
mosaic(data_dict, title="Diagrama de mosaico: Día de la semana vs Momento del día")
plt.show()

"""### Graficos interactivos

#### Dispersión Price vs Quantity
"""

graficar_dispersión_interactiva(
    data=df_sales,
    x="quantity",
    y="price",
    color="mes",
    hover_cols=["item_name", "price", "quantity", "monto_total"],
    titulo="Cantidad vs precio por mes",
    etiquetas={"quantity": "Cantidad", "price": "Price"}
)

"""Este gráfico nos permite identificar, en primer lugar, aquellos ítems con precios elevados o que fueron comprados en grandes cantidades. Por ejemplo, productos como “Ceramic Top Storage Jar” y “Paper Craft Little Birdie” destacan por su alto volumen de unidades vendidas. Asimismo, el gráfico revela por primera vez la presencia del ítem “Amazon Fee”, que se destaca como el producto con el precio unitario más alto, un hallazgo que será analizado en mayor profundidad en etapas posteriores.

Adicionalmente, al observar los ítems de menor precio y cantidad, se detecta una posible correlación inversa entre el precio y la cantidad comprada: a menor precio, mayor volumen de unidades adquiridas. Sin embargo, esta relación no parece ser particularmente fuerte.

#### Evolución mensual por top categorías
"""

df_transacciones_cat=df_sales_date[['week','categoria','bill_no']].drop_duplicates()
transacciones_semanales_cat=df_transacciones_cat.groupby(['week','categoria']).agg({'bill_no':'count'}).reset_index()
transacciones_semanales_cat.rename(columns={'bill_no':'cantidad_transacciones'},inplace=True)

transacciones_semanales_cat_top = transacciones_semanales_cat.loc[transacciones_semanales_cat.categoria.isin(top_cat)]
transacciones_semanales_cat_top

# Create the interactive line plot using Plotly Express
fig = px.line(transacciones_semanales_cat_top,
              x='week',
              y='cantidad_transacciones',
              color='categoria',
              title='Evolución Semanal de Cantidad de Transacciones por Categoría',
              labels={'week': 'Semana', 'quantity': 'Cantidad de Transacciones', 'categoria': 'Categoría'})

fig.show()

"""Este gráfico muestra la evolución del total de transacciones por categoría, información que ya fue analizada previamente en el estudio. La versión interactiva permite filtrar por mes y acceder a detalles adicionales mediante el tooltip, lo que facilita la exploración puntual de datos específicos.

Sin embargo, más allá de las mejoras en la visualización, el gráfico no aporta nuevos insights significativos respecto a los análisis ya realizados hasta este punto.

### DASH

Creamos la siguiente visualización para poder mostrar al cliente. Para su acceso, ejecutarlo y acceder al link provisto.

Ej: Tu app está disponible en: NgrokTunnel: "https://grupo3fundamentosdeprogramacion.ngrok.app" -> "http://localhost:8050"

Acceder a https://grupo3fundamentosdeprogramacion.ngrok.app
"""

# # Agregamos nuestro token
# !ngrok config add-authtoken 2wQaOiDah1w1MIpBB4uKeZjqekx_7NpRXFLcznpuSgWju7bFu
# # Dataset simulado
# df = df_sales

# def graficar_doble_eje(
#     data1, x1, y1, label_y1, color1,
#     data2, y2, label_y2, color2,
#     titulo='Evolución con dos variables',
#     formato_y1_en_miles=False,
#     formato_y2_en_miles=False
# ):
#     fig = go.Figure()

#     # Eje 1
#     fig.add_trace(go.Scatter(
#         x=data1[x1], y=data1[y1],
#         mode='lines+markers',
#         name=label_y1,
#         marker=dict(color=color1),
#         yaxis='y1'
#     ))

#     # Eje 2
#     fig.add_trace(go.Scatter(
#         x=data2[x1], y=data2[y2],
#         mode='lines+markers',
#         name=label_y2,
#         marker=dict(color=color2),
#         yaxis='y2'
#     ))

#     fig.update_layout(
#         title=titulo,
#         xaxis=dict(title=x1.capitalize()),
#         yaxis=dict(
#             title=label_y1,
#             showgrid=False,
#             tickformat="," if formato_y1_en_miles else None
#         ),
#         yaxis2=dict(
#             title=label_y2,
#             overlaying='y',
#             side='right',
#             showgrid=False,
#             tickformat="," if formato_y2_en_miles else None
#         ),
#         legend=dict(x=0.01, y=0.99),
#         height=700,
#         width=1300,
#         title_x=0.5
#     )

#     return fig

# #filtros de grafico

# df_filtrado_boxplot_excluir = df_sales[~df_sales["categoria"].isin(["Papelería y Regalos", "Ornamentos y Figuras", "Otros"])]
# df_filtrado_boxplot_precio = df_sales[df_sales.price < 50]
# df_filtrado_boxplot_quantity = df_sales[df_sales.quantity < 50]
# top_items = df_sales.groupby("item_name")["quantity"].sum().sort_values(ascending=False).reset_index().head(20)
# df_sales_date = df_sales[df_sales.date < '2011-12-01'].copy()
# #ventas_mensuales = df_sales_date.groupby(df_sales_date['mes']).agg({'monto_total': 'sum'}).reset_index()
# ventas_mensuales = df_sales_date.groupby('mes').agg({'monto_total': 'sum'}).reset_index()

# # Transacciones también dependen de df_sales_date, así que va después:

# df_transacciones = df_sales_date[['mes', 'bill_no']].drop_duplicates()
# transacciones_mensuales = df_transacciones.groupby('mes').agg({'bill_no': 'count'}).reset_index()
# transacciones_mensuales.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)


# # Agrupación y ordenamiento
# ventas_categorias = (
#     df_sales.groupby('categoria').agg({'quantity': 'sum'}).reset_index().sort_values(by='quantity', ascending=False)
# )
# # Top 20 categorias
# ventas_categorias_top = ventas_categorias.iloc[:20, :]

# df_productos = df_sales_date[['mes', 'bill_no', 'item_name']].drop_duplicates()
# productos_mensuales = df_productos.groupby(['mes','bill_no']).agg({'item_name': 'count'}).reset_index()
# productos_mensuales.rename(columns={'item_name': 'cantidad_items_distintos'}, inplace=True)

# productos_mensuales_avg = productos_mensuales.groupby('mes').agg({'cantidad_items_distintos': 'mean'}).reset_index()

# df_clientes = df_sales_date[['mes', 'customer_id']].drop_duplicates()
# clientes_mensuales = (df_clientes.groupby('mes').agg({'customer_id': 'count'}).reset_index().rename(columns={'customer_id': 'cantidad_customer_id'}))

# df_compras_clientes = df_sales_date[['mes', 'customer_id', 'bill_no']].drop_duplicates()

# compras_clientes_mensuales = ( df_compras_clientes.groupby(['mes', 'customer_id']).agg({'bill_no': 'count'}).reset_index().rename(columns={'bill_no': 'cantidad_compras_cliente'}))

# # Excluir el customer_id 99999
# compras_clientes_mensuales = compras_clientes_mensuales[compras_clientes_mensuales['customer_id'] != 99999]

# # Promedio mensual por cliente
# compras_clientes_mensuales_avg = (compras_clientes_mensuales.groupby('mes').agg({'cantidad_compras_cliente': 'mean'}).reset_index())
# compras_clientes_mensuales_max = compras_clientes_mensuales.groupby('mes').agg({'cantidad_compras_cliente': 'max'}).reset_index()

# # Las categorías más importantes (ya definidas análisis anterior)
# top_cat = ventas_categorias_top["categoria"].tolist()

# # Extraer transacciones únicas por categoría
# df_transacciones_cat = df_sales_date[['mes', 'categoria', 'bill_no']].drop_duplicates()
# transacciones_mensuales_cat = ( df_transacciones_cat.groupby(['mes', 'categoria']).agg({'bill_no': 'count'}).reset_index().rename(columns={'bill_no': 'cantidad_transacciones'}))

# # Filtrar solo las categorías top
# transacciones_mensuales_cat_top = transacciones_mensuales_cat[transacciones_mensuales_cat['categoria'].isin(top_cat)]

# df_monto_cat = df_sales_date[['mes', 'categoria', 'monto_total']]
# monto_cat_mensuales = ( df_monto_cat.groupby(['mes', 'categoria']).agg({'monto_total': 'sum'}).reset_index().rename(columns={'monto_total': 'monto_total_categoria'}))

# # Seleccionamos las 6 principales categorías
# top_cat = ventas_categorias.head(6).categoria.unique()
# monto_cat_mensuales_top = monto_cat_mensuales[monto_cat_mensuales['categoria'].isin(top_cat)]

# # Asegurar que la columna date sea datetime
# df_sales_date['date'] = pd.to_datetime(df_sales_date['date'])

# # Obtener el lunes de cada semana
# df_sales_date['week'] = df_sales_date['date'] - pd.to_timedelta(df_sales_date['date'].dt.weekday, unit='d')
# df_sales_date['week'] = df_sales_date['week'].dt.strftime('%Y-%m-%d')

# # Agrupar por semana y contar transacciones únicas
# df_transacciones_semana = df_sales_date[['week', 'bill_no']].drop_duplicates()
# transacciones_semanales = (df_transacciones_semana.groupby('week').agg({'bill_no': 'count'}).reset_index().rename(columns={'bill_no': 'cantidad_transacciones'}))

# # Asegurar que date es datetime
# df_sales_date['date'] = pd.to_datetime(df_sales_date['date'])

# # Crear columnas si aún no existen
# df_sales_date["dia_de_semana"] = df_sales_date["date"].dt.day_name()

# # Crear columna 'momento_del_dia' si no existe
# def clasificar_momento(hora):
#     if 6 <= hora < 12:
#         return "Mañana"
#     elif 12 <= hora < 18:
#         return "Tarde"
#     else:
#         return "Noche"

# df_sales_date["hora"] = df_sales_date["date"].dt.hour
# df_sales_date["momento_del_dia"] = df_sales_date["hora"].apply(clasificar_momento)

# # Agrupar y pivotear
# df_heatmap = df_sales_date.groupby(["dia_de_semana", "momento_del_dia"])["monto_total"].sum().reset_index()

# # Ordenar las filas y columnas según lo deseado
# orden_filas = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# orden_columnas = ["Mañana", "Tarde", "Noche"]

# heatmap_pivot = df_heatmap.pivot(index="dia_de_semana", columns="momento_del_dia", values="monto_total").reindex(index=orden_filas, columns=orden_columnas)

# # Asegurar que date es datetime
# df_sales_date['date'] = pd.to_datetime(df_sales_date['date'])

# # Crear columnas necesarias
# df_sales_date["dia_de_semana"] = df_sales_date["date"].dt.day_name()
# df_sales_date["mes"] = df_sales_date["date"].dt.to_period("M").astype(str)

# # Agrupar
# df_heatmap_mes = df_sales_date.groupby(["dia_de_semana", "mes"])["monto_total"].sum().reset_index()

# # Orden personalizado
# orden_filas = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# orden_columnas = ['2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05','2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11']

# # Pivotear
# heatmap_mes = df_heatmap_mes.pivot(index="dia_de_semana", columns="mes", values="monto_total").reindex(index=orden_filas, columns=orden_columnas)

# df_sales['mes'] = pd.to_datetime(df_sales['date']).dt.to_period("M").astype(str)

# df_transacciones_cat = df_sales_date[['week', 'categoria', 'bill_no']].drop_duplicates()

# transacciones_semanales_cat = (df_transacciones_cat.groupby(['week', 'categoria']).agg({'bill_no': 'count'}).reset_index().rename(columns={'bill_no': 'cantidad_transacciones'}))

# # Usamos solo las categorías principales
# transacciones_semanales_cat_top = transacciones_semanales_cat[transacciones_semanales_cat['categoria'].isin(top_cat)]

# heatmap_hora_data = (
#     df_sales.groupby(["dia_de_semana", "hora"])["bill_no"]
#     .nunique()
#     .reset_index(name="cantidad_ventas")
# )

# # Pivot para el heatmap
# orden_filas = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# orden_columnas = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# heatmap_hora = (
#     heatmap_hora_data
#     .pivot(index="dia_de_semana", columns="hora", values="cantidad_ventas")
#     .reindex(index=orden_filas, columns=orden_columnas)
# )

# # Top 20 ítems dentro de la categoría "Otros"
# df_otros = df[df['categoria'].str.lower() == 'otros']
# top_otros = df_otros.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(20).reset_index()

# # App Dash
# app = Dash(__name__)
# app.title = "Dashboard de Ventas"

# app.layout = html.Div([
#     html.H2("Selecciona un tipo de gráfico"),
#     dcc.Dropdown(
#         id="tipo-grafico",
#         options=[
#             {"label": "Boxplot original", "value": "boxplot"},
#             {"label":"Boxplot sin presencia outliers en categorias con mayor presencia", "value":"boxplot_sva"},
#             {"label":"Boxplot filtrado por precio", "value":"boxplot_precio"},
#             {"label":"Boxplot filtrado tambien por cantidad", "value":"boxplot_cantidad"},
#             {"label":"Barras ventas segun categoria", "value":"barras_categorias_20"},
#             {"label":"Barras cantidad segun item", "value":"barras_items"},
#             {"label": "Barras segun Ítems en Categoría Otros", "value": "top_items_otros"},
#             {"label": "Lineal de la Evolucion Mensual del Monto Total de Ventas.","value":"evolucion_mensual"},
#             {"label": "Lineal de Transacciones Mensuales","value":"transacciones_mensuales"},
#             {"label": "Doble eje: Ventas vs Transacciones", "value": "doble_eje"},
#             {"label": "Evolución Mensual de Ítems distintos por factura", "value": "evolucion_items_distintos"},
#             {"label": "Doble eje: Transacciones vs. Ítems distintos por factura", "value": "transacciones_vs_items"},
#             {"label": "Evolución Mensual de Clientes Distintos", "value": "clientes_mensuales"},
#             {"label": "Doble eje: Transacciones vs Clientes", "value": "transacciones_vs_clientes"},
#             {"label": "Evolución Promedio de Compras por Cliente", "value": "promedio_compras_cliente"},
#             {"label": "Doble eje: Compras por Cliente vs Ítems distintos", "value": "compras_vs_items"},
#             {"label": "Evolución de la Máxima Cantidad de Compras por Cliente", "value": "max_compras_cliente"},
#             {"label": "Evolución de Transacciones por Categoría (Top)", "value": "transacciones_categoria_top"},
#             {"label": "Evolución de Monto por Categoría (Top 6)", "value": "monto_categoria_top"},
#             {"label": "Evolución Semanal de Transacciones", "value": "transacciones_semanales"},
#             {"label": "Mapa de Calor de Ventas por Día y Momento", "value": "heatmap_ventas"},
#             {"label": "Mapa de Calor por Día de la Semana y Mes", "value": "heatmap_dia_mes"},
#             {"label": "Mapa de Calor de Ventas por Hora y Día", "value": "heatmap_hora_dia"},
#             {"label": "Dispersión: Cantidad vs Precio por Mes", "value": "dispersión_cantidad_precio"},
#             {"label": "Evolución Semanal de Transacciones por Categoría", "value": "transacciones_semanales_categoria"}



#         ],
#         value="boxplot"
#     ),
#     html.P(id="descripcion"),
#     dcc.Graph(id="grafico")

# ])

# @app.callback(
#     Output("grafico", "figure"),
#     Output("descripcion", "children"),
#     Input("tipo-grafico", "value")
# )
# def renderizar_grafico(tipo):
#     print("📏 transacciones_mensuales:", transacciones_mensuales.shape)
#     print("📏 ventas_mensuales:", ventas_mensuales.shape)
#     if tipo == "boxplot":
#         fig = px.box(df, x="price", y="categoria", title="Precios por categoría", height=900, width=1500, color="categoria" )
#         return fig, "Antes que nada queremos observar la distribución de los precios por categoría para detectar la presencia de outliers y si las distintas categorías se comportan de manera similar."
#     elif tipo == "boxplot_sva":
#         fig = px.box(df_filtrado_boxplot_excluir, x="price", y="categoria", orientation="h", color="categoria", title="Boxplot de Precios por Categoría (excluyendo 3 categorías con outliers)", height=900, width=1500)
#         return fig,"Como en la categorías 'Papelería y Regalos', 'Ornamentos y Figuras' y 'Otros' se detecta la presencia de valores muy altos, se hace difícil la visualización de los boxplot, por lo tanto decidimos excluirlas del gráfico."
#     elif tipo == "boxplot_precio":
#         fig = px.box(df_filtrado_boxplot_precio, x="price", y="categoria", orientation="h", height=900, width=1500, color="categoria" )
#         return fig,"Ahora podemos observar con mayor claridad las diferencias entre las categorías, por ejemplo Panificados tiene una media superior al resto. Si queremos ver aún más en detalle podemos filtrar por precio como se ve en la siguiente gráfica. Algunas categorías se comportan de manera similar, tienen medias y cuantiles parecidos y otras son bien distintas como muebles que por ejemplo tiene un mínimo mucho más alto que el resto."
#     elif tipo == "boxplot_cantidad":
#         fig = px.box(df_filtrado_boxplot_quantity, x="quantity", y="categoria", height=900, width=1500, orientation="h", color="categoria" )
#         return fig,"Como sigue siendo difícil ver las cajas de las medias y cuantiles porque la distribuciones tienen una cola hacia la derecha muy extensa, filtramos también por cantidad. Esto nos permite ver que en relación a la media, el quantil 25 y el 75 hay categorías con comportamientos similares. Muebles parece ser la que tiene valores más bajos con respecto a estas métricas."
#     elif tipo == "barras_categorias_20":
#         fig = px.bar(ventas_categorias_top, x="quantity", y="categoria", height=900, width=1500, orientation="h", color="categoria" )
#         return fig,"En el siguiente gráfico podemos observar que 'Papelería y Regalos', 'Otros' y 'Bolsas y Organizadores' son las que ocupan los primeros lugares con más 500.000 unidades vendidas. La categoría 'Otros' si bien se encuentra en el segundo lugar, agrupa muchos productos por lo tanto es de esperar que esté en los primeros lugares. Los últimos 10 lugares muestran una distribución mucho más uniforme que los primeros 10."
#     elif tipo == "barras_items":
#         fig = px.bar(top_items.sort_values(by="quantity", ascending=True), x="quantity", y="item_name", height=900, width=1500, orientation="h", title="Top 10 Items")
#         return fig,"En este gráfico podemos observar que los primeros cuatro productos tienen una cantidad de ventas muy alta con respecto al resto."
#     elif tipo == "evolucion_mensual":
#         fig = px.line(ventas_mensuales, x="mes", y='monto_total', height=900, width=1500, title='Evolución Mensual del Monto Total de Ventas')
#         return fig,"En este gráfico podemos observar que el monto total de ventas desde diciembre de 2010 hasta agosto de 2011 es bastante estable pero a partir de setiembre empieza a subir considerablemente. En los siguientes gráficos vamos a investigar para tratar de encontrar la razón de esta suba."
#     elif tipo == "transacciones_mensuales":
#         fig = px.line(transacciones_mensuales, x="mes", y='cantidad_transacciones', height=900, width=1500,  title='Evolución Mensual de Cantidad de Transacciones')
#         return fig, "Durante la mayor parte del año, las transacciones se mantienen relativamente estables, pero a partir de septiembre de 2011 se observa un crecimiento acelerado"
#     elif tipo == "doble_eje":
#         fig = graficar_doble_eje(data1=transacciones_mensuales, x1='mes',  y1='cantidad_transacciones', label_y1='Cantidad de Transacciones', color1='red', data2=ventas_mensuales, y2='monto_total', label_y2='Ventas Totales ($)', color2='green', titulo='Evolución Mensual: Ventas Totales y Transacciones', formato_y1_en_miles=False, formato_y2_en_miles=False)
#         return fig, "Este gráfico muestra las ventas totales comparadas con la cantidad de transacciones mensuales en un doble eje."
#     elif tipo == "evolucion_items_distintos":
#         fig = px.line(productos_mensuales_avg, x="mes", y="cantidad_items_distintos", title="Evolución Mensual de Cantidad de Items distintos vendidos", markers=True)
#         fig.update_traces(line=dict(color="blue"), marker=dict(color="blue"))
#         fig.update_layout( height=700, width=1200, title_x=0.5, xaxis_title="Mes", yaxis_title="Promedio de ítems distintos por factura")
#         fig.update_yaxes(tickformat=",")  # Formato en miles
#         return fig, "Este gráfico muestra la evolución del promedio de ítems distintos por factura a lo largo del tiempo. Es útil para analizar la diversidad de productos comprados mes a mes."
#     elif tipo == "transacciones_vs_items":
#         fig = graficar_doble_eje(data1=transacciones_mensuales, x1='mes', y1='cantidad_transacciones', label_y1='Cantidad de Transacciones', color1='red', data2=productos_mensuales_avg, y2='cantidad_items_distintos', label_y2='Promedio de ítems distintos por factura', color2='blue', titulo='Evolución Mensual: Transacciones vs. Diversidad de Ítems por Factura', formato_y1_en_miles=False, formato_y2_en_miles=False)
#         return fig, "Con respecto a la suba de transacciones de los últimos meses si bien ambas variables parecen moverse en el mismo sentido, no observamos que pueda haber una correlación ya que el comportamiento en los meses anteriores es muy disímil. Llama la atención que en Enero de 2011 bajan un poco la cantidad de transacciones pero sube la cantida promedio de items distintos por transacción, esto puede deberse a algún valor atípico de una compra muy grande. Otro mes que llama la atención es mayo de 2011 en el que sucede lo opuesto, suben las transacciones pero la cantidad promedio de items distintos alcanza su punto más bajo."
#     elif tipo == "clientes_mensuales":
#         fig = px.line(clientes_mensuales, x="mes", y="cantidad_customer_id", title="Evolución Mensual de Cantidad de Clientes Distintos", markers=True)
#         fig.update_traces(line=dict(color="violet"), marker=dict(color="violet"))
#         fig.update_layout(height=700, width=1200, title_x=0.5, xaxis_title="Mes", yaxis_title="Cantidad de Clientes" )
#         fig.update_yaxes(tickformat=",")  # Formato en miles
#         return fig, "Este gráfico muestra cómo evoluciona mes a mes la cantidad de clientes únicos que realizaron compras. Es útil para analizar el crecimiento o retención de la base de clientes."
#     elif tipo == "transacciones_vs_clientes":
#         fig = graficar_doble_eje(data1=transacciones_mensuales, x1='mes', y1='cantidad_transacciones', label_y1='Cantidad de Transacciones', color1='red', data2=clientes_mensuales, y2='cantidad_customer_id', label_y2='Cantidad de Clientes', color2='violet', titulo='Evolución Mensual: Transacciones vs. Clientes Distintos', formato_y1_en_miles=False, formato_y2_en_miles=False )
#         return fig, "Dado que la evolución de la cantidad de clientes muestra la misma tendencia, persumimos que esta es la explicación del aumento, sin embargo, vamos a revisarlo en la siguiente gráfica."
#     elif tipo == "promedio_compras_cliente":
#         fig = px.line(compras_clientes_mensuales_avg, x="mes", y="cantidad_compras_cliente",  title="Evolución Mensual de Cantidad de Compras Promedio por Cliente")
#         fig.update_traces(line=dict(color="blue"), marker=dict(color="blue"))
#         fig.update_layout( height=700, width=1200, title_x=0.5,  xaxis_title="Mes", yaxis_title="Promedio de Compras por Cliente", yaxis_range=[1, 2])
#         return fig, "Este gráfico muestra cuántas compras en promedio hizo cada cliente mes a mes, excluyendo a los clientes atípicos como el ID 99999. Es útil para detectar cambios en la frecuencia de compra."
#     elif tipo == "compras_vs_items":
#         fig = graficar_doble_eje( data1=compras_clientes_mensuales_avg, x1='mes', y1='cantidad_compras_cliente', label_y1='Promedio de Compras por Cliente', color1='blue', data2=productos_mensuales_avg, y2='cantidad_items_distintos', label_y2='Promedio de Ítems distintos por factura', color2='purple', titulo='Evolución Mensual: Compras por Cliente vs Diversidad de Productos', formato_y1_en_miles=False, formato_y2_en_miles=False)
#         return fig, "Este gráfico compara cuántas compras realiza cada cliente en promedio con la variedad de productos distintos que se compran por factura. Es útil para ver si los clientes compran más seguido, más variado, o ambas cosas."
#     elif tipo == "max_compras_cliente":
#         fig = px.line(compras_clientes_mensuales_max, x="mes", y="cantidad_compras_cliente", title="Evolución Mensual de Máxima Cantidad de Compras por Cliente")
#         fig.update_traces(line=dict(color="blue"), marker=dict(color="blue"))
#         fig.update_layout( height=700, width=1200, title_x=0.5, xaxis_title="Mes", yaxis_title="Máxima Cantidad de Compras por Cliente", yaxis_range=[0, 50])
#         return fig, "Este gráfico muestra el cliente más comprador de cada mes, es decir, cuántas compras realizó el cliente con mayor frecuencia en ese período. Es útil para identificar posibles mayoristas, revendedores o comportamientos atípicos."
#     elif tipo == "transacciones_categoria_top":
#         fig = px.line( transacciones_mensuales_cat_top, x="mes",y="cantidad_transacciones", color="categoria", title="Evolución Mensual de Cantidad de Transacciones por Categoría (Top)",markers=True)
#         fig.update_layout( height=750, width=1300, title_x=0.5, xaxis_title="Mes", yaxis_title="Cantidad de Transacciones" )
#         fig.update_yaxes(tickformat=",")  # Formato miles
#         return fig, "Si estudiamos la evolución del total de transacciones por el top de categorías vemos un comprtamiento similar entre ellas y similar a la tendencia general."
#     elif tipo == "monto_categoria_top":
#         fig = px.line( monto_cat_mensuales_top, x="mes", y="monto_total_categoria", color="categoria", title="Evolución Mensual de Monto Total de Ventas por Top de Categoría", markers=True)
#         fig.update_layout( height=750, width=1300, title_x=0.5, xaxis_title="Mes", yaxis_title="Monto Total de Ventas")
#         fig.update_yaxes(tickformat=",")  # Formato en miles
#         return fig, "Llama la atención que si bien suben las transacciones el monto total para el resto de las categorías se mantiene, esto puede deberse a una baja de precios o baja en cantidad por items."
#     elif tipo == "transacciones_semanales":
#         fig = px.line(transacciones_semanales, x="week", y="cantidad_transacciones",  title="Evolución Semanal de Cantidad de Transacciones",markers=True )
#         fig.update_traces(line=dict(color="orange"), marker=dict(color="orange"))
#         fig.update_layout(height=700,width=1300,title_x=0.5,xaxis_title="Semana",yaxis_title="Cantidad de Transacciones")
#         fig.update_yaxes(tickformat=",")  # Formato miles
#         return fig, "La evolución semanal de las ventas nos permite observar con mayor detalle cuáles son las semanas en las que hay más ventas, más allá del aumento que vemos en los ultimos meses esto nos permite ver que no es constante sino que en muchos casos puede deberse a promociones que se dan en días concretos. Sucede algo similar la semana del 6 de diciembre de 2010 y el 9 de mayo de 2011."
#     elif tipo == "heatmap_ventas":
#         fig = go.Figure(data=go.Heatmap(z=heatmap_pivot.values,x=heatmap_pivot.columns,y=heatmap_pivot.index,colorscale='Viridis',colorbar=dict(title='Monto Total'),text=heatmap_pivot.values.round(0),texttemplate="%{text:.}", hovertemplate="Día: %{y}<br>Momento: %{x}<br>Monto: %{z:,}<extra></extra>"))
#         fig.update_layout(title="Mapa de calor de monto total de ventas por día de la semana y momento del día",xaxis_title="Momento del día",yaxis_title="Día de la semana",height=600,width=900,title_x=0.5)
#         return fig, "Si bien en este heatmap podemos ver que los Martes en la tarde es el momento en el que se registran mayor monto de transacciones, no podemos saber si es todos los jueves ni se es en todas las tardes ya que esta es información agregada, puede haberse dado en un mes en particular."
#     elif tipo == "heatmap_dia_mes":
#         fig = go.Figure(data=go.Heatmap(z=heatmap_mes.values,x=heatmap_mes.columns,y=heatmap_mes.index,colorscale='Viridis',colorbar=dict(title='Monto Total'),text=heatmap_mes.values.round(0), texttemplate="%{text:.}", hovertemplate="Día: %{y}<br>Mes: %{x}<br>Monto: %{z:,}<extra></extra>"))
#         fig.update_layout(title="Mapa de calor de monto total de ventas por día de la semana y mes",xaxis_title="Mes",yaxis_title="Día de la semana",height=600,width=1000,title_x=0.5)
#         return fig, "Sucede algo similar en este gráfico, los lunes de noviembre de 2011 fueron los días que concentraron mayor monto de ventas pero no podemos concluir mucho más."
#     elif tipo == "heatmap_hora_dia":
#         fig = go.Figure(data=go.Heatmap(
#             z=heatmap_hora.values,
#             x=heatmap_hora.columns,
#             y=heatmap_hora.index,
#             colorscale="Blues",
#             colorbar=dict(title='Cantidad de Ventas'),
#             text=heatmap_hora.values.round(0),
#             texttemplate="%{text:.0f}",
#             hovertemplate="Día: %{y}<br>Hora: %{x}h<br>Ventas: %{z}<extra></extra>"
#         ))
#         fig.update_layout(
#             title="Mapa de Calor de Cantidad de Ventas por Hora y Día",
#             xaxis_title="Hora del Día",
#             yaxis_title="Día de la Semana",
#             height=600,
#             width=1000,
#             title_x=0.5
#         )
#         return fig, "Con el objetivo de identificar patrones temporales en el comportamiento de compra, se construyó un mapa de calor que muestra el monto total de ventas segmentado por día de la semana y momento del día (mañana, tarde y noche). Esta visualización permite detectar de manera rápida los períodos con mayor actividad comercial, así como posibles oportunidades para optimizar promociones, horarios de atención o gestión de recursos."
#     elif tipo == "top_items_otros":
#         fig = px.bar(
#             top_otros.sort_values(by="quantity"),
#             x="quantity",
#             y="item_name",
#             orientation="h",
#             color_discrete_sequence=px.colors.qualitative.Set3
#         )
#         fig.update_layout(
#             title="Top 20 de Ítems más Vendidos en la Categoría 'Otros'",
#             xaxis_title="Cantidad Vendida",
#             yaxis_title="Ítem",
#             height=700,
#             width=1200,
#             title_x=0.5
#         )
#         fig.update_traces(text=top_otros["quantity"].apply(lambda x: f"{x:,}"), textposition='outside')

#         return fig, "Este gráfico presenta los ítems más vendidos dentro de la categoría 'Otros'. Es útil para identificar qué productos aportan más volumen dentro de esta categoría general y si merece la pena desagregarla aún más en futuras segmentaciones."

#     elif tipo == "dispersión_cantidad_precio":
#         fig = px.scatter(df_sales,x="quantity",y="price",color="mes",hover_data=["item_name", "price", "quantity", "monto_total"],title="Cantidad vs Precio por Mes",labels={"quantity": "Cantidad", "price": "Precio"})
#         fig.update_layout(height=750,width=1300,title_x=0.5)
#         return fig, "Este gráfico de dispersión nos permite visualizar primero que nada aquellos items que tienen precios muy altos o los que se compraron en una cantidad muy grande. Por ejemplo, los items 'ceramic top storage jar' y el 'paper craft little birdie' se compraron muchas unidades y el item 'amazon fee' es el que tiene mayor precio. Por otro lado si hacemos zoom en los items con menor precio y cantidad vemos cierta correlación entre precio más bajo y mayor cantidad de items comprados aunque no es muy fuerte.Este gráfico además, al ser interacitvo, nos permite analizar las compras por mes, puediendo detectar en cada uno de ellos, por ejemplo, cuales fueron los items más comprados o con mayor precio."
#     elif tipo == "transacciones_semanales_categoria":
#          fig = px.line(transacciones_semanales_cat_top,x='week',y='cantidad_transacciones',color='categoria',title='Evolución Semanal de Cantidad de Transacciones por Categoría',labels={'week': 'Semana', 'cantidad_transacciones': 'Cantidad de Transacciones', 'categoria': 'Categoría'},markers=True)
#          fig.update_layout(height=750,width=1300,title_x=0.5)
#          return fig, "Este grafico nos muestra que la evolución del total de transacciones para las distintas categorías sigue la misma tendencia, confirmando que las subas son más por promociones generales. El gráfico interactivo nos permite seleccionar los meses que queremos ver además de obtener mayor información en el tooltip."

#     return go.Figure(), "❓ Tipo de gráfico no reconocido."

# # Servidor con threading
# def run():
#     serve(app.server, host="0.0.0.0", port=8050)

# thread = threading.Thread(target=run)
# thread.daemon = True
# thread.start()

# # Esperar a que arranque y conectar ngrok
# time.sleep(5)
# public_url = ngrok.connect(8050, proto="http", domain="grupo3fundamentosdeprogramacion.ngrok.app")
# print("🌐 Tu app está disponible en:", public_url)

import plotly.express as px

fig = px.bar(
    top_items,
    x="quantity",
    y="item_name",
    orientation="h",
    text="quantity",
    title="Top 20 Items más vendidos"
)
fig.update_traces(textposition='outside')
fig.update_layout(height=800, width=1000, title_x=0.5)
fig.show()

"""### **Analisis Visualización de datos**

**Describe las observaciones más relevantes obtenidas a partir de cada uno de los gráficos generados. ¿Qué tendencias, patrones o distribuciones puedes identificar?**

**¿Fue posible representar todas las variables del conjunto de datos mediante histogramas y gráficos de barras? Justifica tu respuesta considerando el tipo de cada variable (categórica o numérica) y la adecuación de cada tipo de gráfico.**

Contamos con variables de tipo categórico, numérico y temporal. Con estos tipos logramos construir gráficas de barra para representar las variables categóricas, boxplot para graficar la distribución de las variables numéricas separando por categorías. Variables temporales nos permitieron realizar graficos de evolución temporal. Algunas variables temporales fueron utilizadas para hacer gráficas de tipo heatmap.

**Entre todos los gráficos generados, ¿cuál(es) consideras que aportan mayor valor informativo? Especifica qué variable(s) se representan en esos gráficos y por qué resultan especialmente útiles para el análisis.**

El análisis comenzo de forma exploratoria viendo la distribución del precio por categoría, y luego llegamos a la evolución mensual del monto total de ventas por mes, que despertó nuestra curiosidad y busqueda de justificación.

Por esta razón, consideramos que las graficas realizadas todas agregan valor, sin contar la de mosaico previamente descartada, ya que nos ayudan a entender el problema.

### **Resultados y Discusión**

Entre todos los gráficos generados, ¿cuál(es) consideras que aportan mayor valor informativo? Especifica qué variable(s) se representan en esos gráficos y por qué resultan especialmente útiles para el análisis.

¿La información obtenida a partir de los gráficos permite responder a alguna de las preguntas clave planteadas en el análisis exploratorio? En caso afirmativo, indica cuál o cuáles preguntas pueden ser abordadas con mayor claridad gracias a las visualizaciones realizadas.

El análisis de las ventas revela diferencias significativas en el comportamiento de las categorías de productos. Algunas, como 'Muebles', tienen precios más altos y distribuciones distintas, mientras que otras como 'Papelería y Regalos' presentan un volumen considerable de unidades vendidas. Se identificaron categorías con outliers que dificultan la visualización, pero al excluirlas, fue posible analizar mejor las tendencias generales. Los productos más vendidos destacan por cantidades notablemente altas, reflejando su popularidad frente al resto.

En términos temporales, el monto total de ventas fue estable durante gran parte del período analizado (diciembre de 2010 a agosto de 2011), con un aumento significativo a partir de septiembre, atribuido principalmente al crecimiento en la cantidad de transacciones podructo de la suba de clientes. Eventos puntuales como promociones podrían explicar estas variaciones, aunque también se notaron fluctuaciones en indicadores como el ticket promedio o la cantidad de ítems por transacción en meses específicos como enero y mayo de 2011.

Al observar los nombres de los ítems y ciertos outliers, se puede inferir que las ventas corresponden a una tienda que ofrece envíos y una amplia variedad de productos, más que a un supermercado tradicional, esto tambien podemos deducirlo por la existencia de un 'Amazon fee'. Productos con precios extremos o volúmenes elevados, como el 'ceramic top storage jar' o el 'paper craft little birdie', refuerzan esta hipótesis y sugieren un negocio diversificado con una oferta muy heterogénea.

Finalmente, la correlación entre precios bajos y cantidades altas fue moderada, con ciertos productos destacándose por extremos en precios o volúmenes vendidos.

Este análisis proporciona información valiosa para identificar patrones de comportamiento del consumidor y ajustar estrategias de precios, inventarios y promociones, optimizando así el desempeño comercial de esta tienda. Si bien no responde las preguntas del inicio, descrubrimos otras más interesantes.



Algunas de ellas son:

1) ¿cuál o cuáles son las variables más destacadas para explicar el nivel de ingresos del negocio, y su evolución a lo largo del período de análisis?
 - Aquí, luego de abordar las dimensiones: cantidad de transacciones, cantidad de items distintos por transacción, ticket promedio, cantidad de clientes, cantidad promedio de compras por cliente, categorías más influyentes, entre otras, destacamos los gráficos que nos muestran cómo la cantidad de transacciones y la cantidad de clientes evolucionan de manera casi idéntica al monto total de ventas, lo cual, en conjunto con la gráfica que nos muestra que el promedio de compras por cliente se mantiene estable, nos brindan información valiosa respecto de los siguientes puntos:
      - La cantidad de clientes es una variable fundamental para explicar el ingreso del comercio, no así la cantidad promedio de compras por cliente. Esto significa que los esfuerzos que el comercio realice en aumentar su base de clientes se ven reflejados directamente en sus ingresos. Sin embargo, la base de clientes potenciales es finita, por lo cual es necesario trabajar en la fidelidad del cliente (aumentar la cantidad promedio de compras por cliente).
      - La cantidad de transacciones tambien resulta clave a la hora de analizar las ventas totales, contrario a lo que vemos al estudiar el ticket promedio, o la cantidad de items distintos por transacción.
      De esto se desprende que hasta el momento, el comercio depende de generar mayor cantidad de ventas para obtener mayores ingresos, sin embargo, es necesario explorar la posibilidad de aumentar los ingresos en base al aumento del ticket promedio (preferiblemente vía aumento de cantidades por transacción, y no por aumento de precios).

2) Qué características muestra el comportamiento de compra de los clientes del comercio?
  - Anteriormente mencionamos las pocas transacciones promedio, que muestran un cliente no fidelizado.
  - A pesar de esto, el gráfico con el top 3 de clientes nos muestra un pequeño grupo de clientes que se mantiene en el top de compradores del comercio. Esto representa un activo que la compañía puede utilizar a su favor, con acciones comerciales que apunten a que estos consumidores puedan referir a otros, haciendo que sus clientes más fieles sean quienes promocionen el negocio a través de su propia experiencia, tanto para fidelizar clientes que hoy compran esporádicamente, como para atraer nuevos clientes.


3) Cuál o cuáles categorías son las más representativas a la hora de analizar la evolución de las ventas?
  - Vimos que por cantidad de transacciones todas se comportan similar, pero al analizar el monto por transacción, en la gráfica "Evolución Mensual de Monto Total de Ventas por Top de Categoría" la categoría Papelería y Regalos es la que realmente inclina la balanza. Especialmente, en el período de mayor crecimiento de los ingresos, es la que mayor peso relativo tomó, lo cual es lógico debido a la fechas en las que se dio el aumento (época festiva). Por esta razón sería conveniente que el comercio aproveche las fechas donde mayor visibilidad puede darle a esta categoría, para potenciar el efecto que ésta tiene sobre las ventas totales.

  
4) Que tan receptiva es la demanda a los esfuerzos publicitarios del negocio?
  - Puntualmente viendo la concentración de ventas en un único día en el mapa de calor de noviembre, y asumiendo que no puede ser trivial que esto haya sucedido, sino que tiene que estar relacionado con una acción por parte de la empresa, entendemos que invertir en campañas publicitarias y acciones comerciales genera un gran efecto sobre los ingresos para este comercio, lo cual representa un campo muy fértil para llevar a cabo las recomendaciones que mencionabamos más arriba, ya que es de esperar que representen un retorno significativo.

En base a estas preguntas, y las respuestas derivadas del análisis de los datos, entendimos que el negocio cuenta con los siguientes recursos:

- Poder potenciador de las categorías "estrella"
- Existencia de grupo de clientes muy fieles
- Demanda altamente receptiva a acciones comerciales.

y que puede utilizarlos a su favor para alcanzar el objetivo de aumentar el ingreso, a través de las siguientes dos estrategias:

- Aumentar ticket promedio
- Aumentar frecuencia de compra


Por lo tanto, elaboramos una serie de recomendaciones que podría adoptar el comercio para aumentar sus ingresos en el futuro, con fundamento en el comportamiento pasado:

- Enfocarse en aumentar la fidelidad de los clientes más que en la cantidad de clientes. - Atraer un cliente es más caro que fidelizar los existentes. Un aumento simultáneo de la cantidad de compras de los clientes actuales, y del ticket promedio de los mismos, genera un doble efecto sobre los ingresos percibidos. Posibles acciones: 3x2 u otras en igual dirección; % descuento en próxima compra, etc.
- Fortalecer la estrategia comercial con foco en las categorías más destacadas: enfocar publicidad especialmente en fechas específicas, reforzar inventario, etc.
- Apalancarse en los consumidores más frecuentes para que más clientes "sean como ellos" (compren más y más seguido). A través de acciones como "referidos", testimonios, etc.

Dada la alta receptividad de la demanda, consideramos que estas acciones surtirán el efecto deseado en un tiempo muy breve.

## Análisis de estadísticos, outliers y normalidad

### Cálculos de estadísticos

En todo análisis de datos es fundamental calcular los estadísticos generales y estudiar los outliers. En caso de que en un futuro quisieramos construir un modelo para predecir las ventas por ejemplo, es importante realizar test de normalidad para estar seguros qué modelos podemos aplicar a nuestros datos.

En nuestro dataset contamos solo con dos variables numéricas, cantidad y precio por lo que los estadísticos se calcularan para estas dos.

Lo que podemos observar es que en ambas variables la media es considerablemente mayor que la mediana, esto sugiere una distribución asimétrica sesgada a la derecha, valores positivos extremadamente grandes. Algo de esto lo pudimos visualizar en los boxplot por categoría pero ahora lo confirmamos de manera empírica. En quantity el promedio es casi 10 mientras que la mediana es 3, es decir el 50% de los valores son menores que 3, los valores extremos son los que hacen que el promedio sea mayor. Sucede algos similar en precio pero no tan extremo.

El rango intercuantil de quantity es 9, mirando el Q1 y Q3 sabemos que el 50% de los datos se concentran entre 1 y 10. Price tiene un precio mucho menor, lo que indica una mayor concentración, el 50% de los datos está entre 1.25 y 4.

Si miramos los máximos vemos que hay valores extremadamente altos, los cuales habíamos detectado de manera visual. Esto nos da indicios de que pueden llegar a haber una cantidad de outliers que es posible que tengamos que eliminar porque nos están ensuciando nuestro análisis.


Si pensamos en que nuestro dataset corresponde a compras, puede llegar a pasar que se compre una cantidad elevada de productos, pero 80995 no parece ser muy razonable, esto lo estudiaremos más adelante cuando veamos los outliers. Lo mismo con el precio, la tienda parece vender a precios razonables, es posible que tenga algún producto más exclusivo pero tendemos a pensar que tiende a vender productos de cierto rango de precio, sobre todo para mantener a sus clientes.
"""

num_variables = ['quantity', 'price']
df_sales_num = df_sales_date[num_variables]

calcular_estadisticos(df_sales_num)

"""Otra forma de visualizar los estadìsticos es realizar un gráfico de dispersión y un boxplot, dada la  cantidad de datos que tenemos para poder realizar la visualización extrajimos una muestra aleatoria que es representativa de la totalidad de los datos. Tal como esperábamos los valores extremos no nos permiten visualizar la distribución pero nos confirma una vez más la presencia de outliers que deben ser eliminados."""

# Dada la gran cantidad de datos, se dificulta su visualización real
# Sacamos una muestra de df_sales_date para hacer los gráficos
df_sales_sample = df_sales_date.sample(frac=.01, random_state=42)
df_sales_sample.shape

for columna in df_sales_num.columns:
    visualizar_distribucion(df_sales_sample[num_variables], columna)

"""Sucede algo similar con el Q-Q plot que es una forma visual de detectar normalidad en los datos, los valores extremos no premiten que haya normalidad, habría que estudiar qué sucede luego de la eliminación de los outliers."""

visualizar_hist_and_qq(df_sales_sample[num_variables], 2)

"""### Eliminación de Outliers

Todo lo realizado anteriormente nos dio indicios de la presencia de outliers pero ahora lo confirmaremos y calcularemos cuántos son, utlizando métodos estadísticos, estos nos permiten poder hacerlo sin tener que tomar una decisión aleatoria.

A continuación observaremos el resultado de dos métodos uno de ellos es la utilización del rango intercuartil y el otro es el z-score.
- En el IQR el k nos indica por cuanto multiplicamos el rango, y por lo tanto el límite inferior y superior.
- El Z socre con Z=1, 2 y 3, este Z nos indica a cuanta desviación estandar de la media detecta los outliers. Por ejemplo con Z=2 detectamos los valores que están a 2 desviaciones estandar de la media, el rango abarca aproximadamente el 95% de los datos.

#### Resultados Generales
"""

# Bucle para aplicar las funciones a todas las columnas numéricas
for columna in df_sales_num.columns:
    print(f"\n ---------- IQR -----------")
    print(f"\nDetección de outliers en {columna}:")
    print(f"\nIQR K = 3")
    outliers_iqr = detectar_outliers_iqr(df_sales_num, columna, 3)
    print(f"\nIQR K = 2")
    outliers_iqr = detectar_outliers_iqr(df_sales_num, columna, 2)
    print(f"\nIQR K = 1.5")
    outliers_iqr = detectar_outliers_iqr(df_sales_num, columna, 1.5)

for columna in df_sales_num.columns:
    print(f"\n ---------- ZSCORE -----------")
    print(f"\nZSCORE Z = 2")
    outliers_zscore = detectar_outliers_zscore(df_sales_num, columna, 2)
    print(f"\nZSCORE Z = 3")
    outliers_zscore = detectar_outliers_zscore(df_sales_num, columna)

"""El método IQR es más robusto cuando estamos frente a una variable que no sigue una distribución normal pero, en este caso, tiende a detectar una mayor cantidad de outliers en nuestras dos variables, entre un 7 y 19%. Creemos que es una cantidad muy alta por lo tanto optamos por quedarnos con los resultados del Z-Score. Dado el significado de nuestras variables, no tenemos como saber si el precio que detecta como outlier es efectivamente un valor muy extremo o si puede ser un valor real, lo mismo con la cantidad, es por eso que optamos por el método más conservador.

#### Eliminación de outliers en price

Para realizar la eliminación de los outliers es importante cuantificar los datos antes y después para estar seguros de no estar eliminando información importante, por lo tanto vamos a proceder a:
- Calcular datos generales del dataframe original
- Detectar los outliers y guardar el dataframe sin ellos
- Volver a calcular los estadísticos
- Revisar los datos eliminados
"""

quantity_txt='quantity'
price_txt='price'

cantidad_bill_no = df_sales_date["bill_no"].nunique()
cantidad_customer_id= df_sales_date["customer_id"].nunique()
print(f'Cantidad de filas existentes: {df_sales_date.shape[0]}')
print(f'Cantidad de bill_no existentes: {cantidad_bill_no}')
print(f'Cantidad de customer_id existentes: {cantidad_customer_id}')

detectar_outliers_zscore(df_sales_date, price_txt, 2)
datos_sin_outliers_p = z_score_filter(df_sales_date, price_txt)
print(f'Cantidad de filas existentes despues de remover outliers: {datos_sin_outliers_p.shape[0]}')

cantidad_bill_no_o = datos_sin_outliers_p["bill_no"].nunique()
cantidad_customer_id_o= datos_sin_outliers_p["customer_id"].nunique()
print(f'Cantidad de filas existentes sin outliers: {datos_sin_outliers_p.shape[0]}')
print(f'Differencia de filas posterior a remover outliers: {df_sales_date.shape[0] - datos_sin_outliers_p.shape[0]} \n')
print(f'Cantidad de bill_no existentes sin outliers: {cantidad_bill_no_o}')
print(f'Differencia bill_no posterior a remover outliers: {cantidad_bill_no - cantidad_bill_no_o} \n')
print(f'Cantidad de customer_id existentes sin outliers: {cantidad_customer_id_o}')
print(f'Differencia de customer_id posterior a remover outliers: {cantidad_customer_id - cantidad_customer_id_o}')

calcular_estadisticos(datos_sin_outliers_p[num_variables])

"""Miramos los datos eliminados"""

df_sales_price_filtered=df_sales_date[df_sales_date['price'] > 95.38].sort_values(by='price', ascending=False)
df_sales_price_filtered.head(6)

df_sales_price_filtered.shape

"""Vemos de que CustomerId hemos sacado información"""

df_sales_price_filtered.customer_id.value_counts().sort_values(ascending=False).head(10)

"""Y de que BillNo"""

df_sales_price_filtered.bill_no.value_counts().sort_values(ascending=False).head(10)

"""En price hay 743 compras con un precio mayor a 95.38 que es el valor del Z-score. Si ordenamos por mayor price se observan muchos cuyo CustomerId es el 99999. Esto nos hace pensar si no deberíamos excluir todos estos registros porque quizás ensucien el dataset, más ahora que sospechamos que son compras online (por la existencia de un amazon fee), lo cual hace que sea más dificil que el comprador no tenga un id.

Si bien se eliminan 743 registros, en términos de bill_no solo se pierden 56 de los 17272 y 7 clientes de los 3887.

####Eliminación de outliers en quantity

Una vez elminados los outliers en price, eliminaremos los de quantity, puede que algunos de ellos se hayan eliminado en el procedimiento anterior.

Tal como con price, vamos a:
- Calcular datos generales del dataframe original
- Detectar los outliers y guardar el dataframe sin ellos
- Volver a calcular los estadísticos
- Revisar los datos eliminados
"""

detectar_outliers_zscore(datos_sin_outliers_p[num_variables], quantity_txt, 3)
datos_sin_outliers_q = z_score_filter(datos_sin_outliers_p, quantity_txt)
print(f'Cantidad de filas existentes despues de remover outliers: {datos_sin_outliers_q.shape[0]}')

print(f'Datos previos a eliminar quantity outliers\n')
print(f'Cantidad de filas existentes: {df_sales_date.shape[0]}')
print(f'Cantidad de filas existentes sin price outliers: {datos_sin_outliers_p.shape[0]}')
print(f'Differencia de filas posterior a remover price outliers: {df_sales_date.shape[0] - datos_sin_outliers_p.shape[0]} \n')
print(f'Cantidad de bill_no existentes: {cantidad_bill_no}')
print(f'Cantidad de bill_no existentes sin price outliers: {cantidad_bill_no_o}')
print(f'Differencia bill_no posterior a remover price outliers: {cantidad_bill_no - cantidad_bill_no_o} \n')
print(f'Cantidad de customer_id existentes: {cantidad_customer_id}')
print(f'Cantidad de customer_id existentes sin price outliers: {cantidad_customer_id_o}')
print(f'Differencia de customer_id posterior a remover price outliers: {cantidad_customer_id - cantidad_customer_id_o}')

cantidad_bill_no_oq = datos_sin_outliers_q["bill_no"].nunique()
cantidad_customer_id_oq = datos_sin_outliers_q["customer_id"].nunique()
print(f'Sin outliers de precio y quantity\n')
print(f'Cantidad de filas existentes: {datos_sin_outliers_q.shape[0]}')
print(f'Cantidad de bill_no existentes: {cantidad_bill_no_oq}')
print(f'Cantidad de customer_id existentes: {cantidad_customer_id_oq}')

print(f'Diferencias al remover quantity outliers')
print(f'Differencia de filas posterior a remover outliers: {datos_sin_outliers_p.shape[0] - datos_sin_outliers_q.shape[0]}')
print(f'Differencia bill_no posterior a remover outliers: {cantidad_bill_no_o - cantidad_bill_no_oq}')
print(f'Differencia de customer_id posterior a remover outliers: {cantidad_customer_id_o - cantidad_customer_id_oq}')

print(f'Diferencias al remover price y quantity outliers')
print(f'Differencia de filas posterior a remover outliers: {df_sales_date.shape[0] - datos_sin_outliers_q.shape[0]}')
print(f'Differencia bill_no posterior a remover outliers: {cantidad_bill_no - cantidad_bill_no_oq}')
print(f'Differencia de customer_id posterior a remover outliers: {cantidad_customer_id - cantidad_customer_id_oq}')

calcular_estadisticos(datos_sin_outliers_q[num_variables])

"""Miramos los datos eliminados"""

df_sales_quantity_filtered=df_sales_date[df_sales_date['quantity'] > 352].sort_values(by='quantity', ascending=False)
df_sales_quantity_filtered.head(6)

df_sales_quantity_filtered.customer_id.value_counts().sort_values(ascending=False).head(10)

df_sales_quantity_filtered.sort_values(by='quantity').quantity.describe()

datos_sin_outliers=datos_sin_outliers_q.copy()

"""En la variable quantity, se identificaron 655 compras con cantidades superiores a 352. Al revisar los item_name correspondientes, se observa que estos tienen coherencia comercial, y no se detecta una concentración significativa en el customer_id 99999, que había sido imputado en casos donde el identificador original era nulo. Esto sugiere que los valores atípicos en quantity podrían corresponder a compras reales, posiblemente asociadas a eventos puntuales o compras mayoristas.

Tras la eliminación de outliers en las variables price y quantity, se perdieron 193 bill_no y 21 customer_id del conjunto de datos.

### Visualización - Dataframe sin Outliers
"""

cantidad_bill_no = df_sales_date["bill_no"].nunique()
cantidad_customer_id= df_sales_date["customer_id"].nunique()
print(f'Cantidad de filas existentes en dataframe original: {df_sales_date.shape[0]}')
print(f'Cantidad de bill_no existentes en dataframe original: {cantidad_bill_no}')
print(f'Cantidad de customer_id existentes en dataframe original: {cantidad_customer_id}')

cantidad_bill_no = datos_sin_outliers["bill_no"].nunique()
cantidad_customer_id= datos_sin_outliers["customer_id"].nunique()
print(f'Cantidad de filas existentes sin outliers de precio y quantity: {datos_sin_outliers.shape[0]}')
print(f'Cantidad de bill_no existentes sin outliers de precio y quantity {cantidad_bill_no}')
print(f'Cantidad de customer_id existentessin outliers de precio y quantity: {cantidad_customer_id}')

"""Una vez elminados los outliers creemos importane volver a realizar algunas visualizaciones, tanto el gráfico de distribución, como el dispersión y algunas gráficas de evolución temporal para comprobar si aquellos picos que observábamos siguen estando.

#### Gráficos de distribución

Queremos observar ahora como cambió la distribución de nuestras variables numéricas luego de la eliminación de los outliers.

Si bien ahora se puede observar un poco mejor la distribución, hay una concentración mayor en los valores más pequeños y los valores más altos tienen menos registros. Esto es de esperar dada la naturaleza de los datos, la tienda tiene muchos artículos a precios bajos, que es lo que atrae al cliente y algunos de precios altos que son especiales. Con respecto a cantidad sucede lo mismo, en general se compran poca cantidad pero puede haber algunos clientes o en ciertos momentos que se compra una mayor cantidad pero son los menos.
"""

for columna in df_sales_num.columns:
    visualizar_distribucion(datos_sin_outliers[num_variables], columna)

visualizar_hist_and_qq(datos_sin_outliers[num_variables], 2)

df_sales_non_outliers=datos_sin_outliers.copy()

cantidad_bill_no = df_sales_non_outliers["bill_no"].nunique()
cantidad_customer_id= df_sales_non_outliers["customer_id"].nunique()
print(f'Cantidad de filas existentes sin outliers de precio y quantity: {df_sales_non_outliers.shape[0]}')
print(f'Cantidad de bill_no existentes sin outliers de precio y quantity {cantidad_bill_no}')
print(f'Cantidad de customer_id existentessin outliers de precio y quantity: {cantidad_customer_id}')

"""Grafico previa eliminación de outliers"""

graficar_dispersión_interactiva(
    data=df_sales_date,
    x="quantity",
    y="price",
    color="mes",
    hover_cols=["item_name", "price", "quantity", "monto_total"],
    titulo="Cantidad vs precio por mes",
    etiquetas={"quantity": "Cantidad", "price": "Price"}
)

"""Grafico posterior eliminación de outliers"""

graficar_dispersión_interactiva(
    data=df_sales_non_outliers,
    x="quantity",
    y="price",
    color="mes",
    hover_cols=["item_name", "price", "quantity", "monto_total"],
    titulo="Cantidad vs precio por mes sin outliers",
    etiquetas={"quantity": "Cantidad", "price": "Price"}
)

"""El gráfico de dispersión muestra claramente el cambio, ahora se puede apreciar más facilmente la disperión y correlación (si es que existe) entre precio y cantidad. Además nos premite estudiar cada producto por separado.

####Gráficos de evolución mensual

#####Ventas

Veremos ahora que sucede con algunos gráficos de evoución mensual, queremos chequear cuánto cambió.
"""

print(f'La diferencia de monto total luego de eliminar los outliers es: {round(df_sales_date.monto_total.sum() - df_sales_non_outliers.monto_total.sum())}')

ventas_mensuales = df_sales_date.groupby(df_sales_date['mes']).agg({'monto_total': 'sum'})
ventas_mensuales_out = df_sales_non_outliers.groupby(df_sales_non_outliers['mes']).agg({'monto_total': 'sum'})

graficar_evolucion_superpuesta(
    data1=ventas_mensuales_out,  # Primer dataset
    data2=ventas_mensuales,  # Segundo dataset
    columna_x='mes',  # Columna para el eje X
    columna_y='monto_total',  # Columna para el eje Y
    titulo='Evolución Mensual de Ventas sin Outliers',
    label_y='Monto Total',
    formato_y_en_miles=True,
    label1='Ventas sin Outliers',
    label2='Ventas con Outliers',
    color1='blue',
    color2='red'
)

graficar_doble_eje_y(
    data1=transacciones_mensuales,
    x1='mes',
    y1='cantidad_transacciones',
    label_y1='Cantidad de Transacciones',
    color1='red',

    data2=productos_mensuales_avg,
    y2='cantidad_items_distintos',
    label_y2='Promedio de items distintos por Bill_no',
    color2='blue',

    titulo='Evolución Mensual: Ventas Totales y Transacciones'
)

"""Si bien se observa una baja en las ventas totales es general, esto nos confirma que los datos atípicos estaban dispersos, no se concentran en un solo día, es decir, no es un error puntual.

#####Transacciones
"""

print(f'La diferencia de transacciones luego de eliminar los outliers es: {round(df_sales_date.bill_no.nunique() - df_sales_non_outliers.bill_no.nunique())}')

transacciones_mensuales = df_sales_date.groupby(df_sales_date['mes']).agg({'bill_no': 'nunique'})
transacciones_mensuales.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)

transacciones_mensuales_out= df_sales_non_outliers.groupby(df_sales_non_outliers['mes']).agg({'bill_no': 'nunique'})
transacciones_mensuales_out.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)

graficar_evolucion_superpuesta(
    data1=transacciones_mensuales_out,
    data2=transacciones_mensuales,
    columna_x='mes',
    columna_y='cantidad_transacciones',
    titulo='Evolución de Cantidad de transaccioness sin Outliers',
    label_y='Monto Total',
    formato_y_en_miles=True,
    label1='Cantidad de transacciones sin Outliers',
    label2='Cantidad de transacciones',
    color1='blue',
    color2='red'
)

"""Luego de la eliminación de outliers al observar la evolución mensual de transacciones no detectamos casi cambios, esto puede deberse a que se eliminaron items en particular dentro de una misma compra (bill_no), y no compras completas, esto se debe a que los clientes tienden a comprar diversa cantidad de items.

#####Clientes
"""

print(f'La diferencia de clientes luego de eliminar los outliers es: {round(df_sales_date.customer_id.nunique() - df_sales_non_outliers.customer_id.nunique())}')

clientes_mensuales = df_sales_date.groupby(df_sales_date['mes']).agg({'customer_id': 'nunique'})
clientes_mensuales.rename(columns={'customer_id': 'clientes'}, inplace=True)

clientes_mensuales_out= df_sales_non_outliers.groupby(df_sales_non_outliers['mes']).agg({'customer_id': 'nunique'})
clientes_mensuales_out.rename(columns={'customer_id': 'clientes'}, inplace=True)

graficar_evolucion_superpuesta(
    data1=clientes_mensuales_out,
    data2=clientes_mensuales,
    columna_x='mes',
    columna_y='clientes',
    titulo='Evolución de Clientes sin Outliers',
    label_y='Monto Total',
    formato_y_en_miles=True,
    label1='Cantidad de Clientes sin Outliers',
    label2='Cantidad de Clientes',
    color1='blue',
    color2='red'
)

"""Sucede algo similar en relación a la cantida de clientes.

En donde más se observa la diferencia es en monto total porque el impacto de los outliers fue en las columnas precio y cantidad pero al no estar concentrados en una fecha, cliente o transacción particular el cambio es más armónico.

##### Ticket promedio
"""

# Calcular monto total por factura (por si no está correctamente consolidado)
# Esto suma los montos por cada bill_no (factura)
facturas = df_sales_non_outliers.groupby(['bill_no', 'dia_del_mes', 'week', 'mes'])['monto_total'].sum().reset_index(name='total_factura')

# Ticket promedio por día
ticket_diario = facturas.groupby(['dia_del_mes', 'mes'])['total_factura'].mean().reset_index(name='ticket_promedio_dia')

# Ticket promedio por semana
ticket_semanal = facturas.groupby('week')['total_factura'].mean().reset_index(name='ticket_promedio_semana')

# Ticket promedio por mes
ticket_mensual = facturas.groupby('mes')['total_factura'].mean().reset_index(name='ticket_promedio_mes')

graficar_doble_eje_y(
    data1=ticket_mensual,
    x1='mes',
    y1='ticket_promedio_mes',
    label_y1='Ticket Promedio',
    color1='magenta',
    limites_y1=(250, 550),

    data2=ventas_mensuales,
    y2='monto_total',
    label_y2='Monto Total',
    color2='green',
    formato_y2_en_miles=True,

    titulo='Evolución Mensual: Ticket Promedio y Ventas'
)

graficar_doble_eje_y(
    data1=ticket_mensual,
    x1='mes',
    y1='ticket_promedio_mes',
    label_y1='Ticket Promedio',
    color1='magenta',
    limites_y1=(250, 550),

    data2=transacciones_mensuales_out,
    y2='cantidad_transacciones',
    label_y2='Cantidad de Transacciones',
    color2='red',

    titulo='Evolución Mensual: Ticket Promedio y Transacciones'
)

"""Queríamos entender cómo se comporta el ticket promedio y observamos que:
en los primeros meses del año el ticket promedio baja un poco pero no tanto como la cantidad de transacciones, se mantiene un poco más estable
luego el ticket promedio empieza a subir pero no acompaña la subida abrupta del mes de agosto
estas dos cosas nos indican que el aumento del monto total está explicado más que nada por la cantidad de transacciones
Es posible que la tienda haya implementado estrategias como promociones o campañas de captación de clientes que incrementaron significativamente el volumen de ventas, incluso si el ticket promedio subió a un ritmo más lento.

##### Top de categorías
"""

top_cat = ventas_categorias.head(6).categoria.unique()
top_cat

# Gráfico de evolución mensual
graficar_xy(
    data=monto_cat_mensuales_top,
    columna_x='mes',
    columna_y='monto_total_categoria',
    titulo='Evolución Mensual de Monto Total de Ventas por Top de Categoría',
    label_y='Monto Total de Ventas',
    formato_y_en_miles=True,
    hue='categoria'
)

transacciones_mensuales_cat_out = df_sales_non_outliers.groupby(['mes', 'categoria']).agg({'bill_no': 'nunique'}).reset_index()
transacciones_mensuales_cat_out.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)
transacciones_mensuales_cat_top_out = transacciones_mensuales_cat_out.loc[transacciones_mensuales_cat_out.categoria.isin(top_cat)]

# Gráfico de evolución mensual sin outliers
graficar_xy(
    data=transacciones_mensuales_cat_top_out,
    columna_x='mes',
    columna_y='cantidad_transacciones',
    titulo='Evolución Mensual de Cantidad de Transacciones por Top de Categoría, sin outliers.',
    label_y='Cantidad de Transacciones',
    color='blue',
    hue='categoria'
)

"""Luego de la eliminación de los oultiers podemos observar que si bien papelería y regalos es la categoría con mayores transacciones, todas siguen la misma tendencia, esto reafirma que no es una categoía en particular que provoca el aumento de las ventas. Confirmamos entonces que el aumento se debe más que nada a la cantidad de clientes. La tienda puedo haberse vuelto más conocida o quizás realizó alguna campaña para nuevos clientes, esta es una buena estrategia a seguir, fidelizar a los clientes que ya tiene y atraer a nuevos."""

top_items = df_sales_non_outliers.groupby("item_name")["bill_no"].nunique().sort_values(ascending=False).reset_index().head(15)
top_items

graficar_barras_horizontales_con_valores(
    data=top_items,
    columna_x="bill_no",
    columna_y="item_name",
    titulo="Cantidad de Transaccioens por Top 15 de Item",
    xlabel="Cantidad de Tranasacicones",
    ylabel="Item"
)

top_items = df_sales_non_outliers.groupby("item_name")["quantity"].sum().sort_values(ascending=False).reset_index().head(15)
top_items

graficar_barras_horizontales_con_valores(
    data=top_items,
    columna_x="quantity",
    columna_y="item_name",
    titulo="Cantidad de Ventas por Top 15 de Item",
    xlabel="Cantidad de Ventas",
    ylabel="Item"
)

"""##### Guardamos el dataframe sin Outliers"""

nombre_archivo= 'df_sales_out.csv'
df_sales_non_outliers.to_csv(ruta + nombre_archivo, index=False)

drive.mount('/content/drive')
ruta = '/content/drive/My Drive/Notebooks/'

nombre_archivo = 'df_sales_out.csv'
df_sales = pd.read_csv(ruta + nombre_archivo, on_bad_lines=lambda line: print(f"Saltando línea: {line}"), engine='python',sep=',')
df_sales.head()

"""### Analisis eliminación Outliers

En nuestro análisis de las variables cantidad y precio, observamos una marcada influencia de valores extremos que generan una asimetría hacia la derecha en ambas distribuciones. Esto se refleja en que las medias son considerablemente mayores que las medianas, y se confirma con los boxplots y el análisis de estadísticos como el rango intercuartílico.

La eliminación de outliers plantea una dificultad importante porque no contamos con información sobre los límites operativos de la tienda:

En **cantidad**: Aunque es razonable suponer que pueden ocurrir compras de productos en grandes cantidades, valores como 80.995 parecen poco probables en una tienda de este tipo. Sin embargo, la naturaleza de los productos vendidos no está clara, lo que dificulta decidir si estos valores son anomalías reales o casos atípicos válidos pero suponemos que si lo son.

En **precio**: Aunque la mayoría de los productos parecen tener precios en un rango razonable, los valores mayores a 95.38 son sospechosos. Al revisar estos registros, encontramos una posible concentración de valores extremos en clientes con un ID imputado (99999), asi como registros de Amazon fee, postage, dotcom postage, que logramos verlo en la visualizacion interactiva, lo que refuerza la posibilidad de datos sucios, ajustes. Sin embargo, desconocemos si la tienda también podría manejar productos exclusivos, o cómo ingresa los cobros de envíos que justifiquen estos precios altos.

**Impacto tras la eliminación de outliers**

Ventas mensuales: Aunque se observa una ligera disminución en el total de ventas, esta no afecta la tendencia general. Esto indica que los outliers no están concentrados en un período específico ni en un cliente puntual, sino que están distribuidos a lo largo del tiempo.

Clientes y transacciones: La eliminación de outliers afecta mínimamente la cantidad total de clientes y transacciones, esto indica que los valores atípicos no están concentrados en determinados clientes ni asociados a un patrón de compra sistemático. Es decir, los outliers parecen ser eventos aislados más que comportamientos recurrentes de ciertos usuarios.

**Consideraciones finales**

Tomar decisiones sobre la eliminación de outliers en este caso fue un desafío porque desconocemos los límites reales del negocio pero los vamos a tomar como atípicos porque la eliminación de outliers mejora la visualización y facilita el análisis de tendencias generales, pero podría llevarnos a descartar datos válidos si no comprendemos el contexto real del negocio.


Es clave diferenciar entre "ruido" y eventos reales: si algunas de esas transacciones eran legítimas, podrían representar oportunidades comerciales valiosas.

El análisis sin outliers elimina sesgos, pero también puede subestimar ingresos máximos o comportamientos de clientes premium.

Idealmente, se puede realizar un análisis con y sin outliers: uno enfocado en la tendencia general, y otro para explorar oportunidades en las ventas de alto impacto.

Acotaremos nuestro estudio a un enfocoque de tendencia general.

### Test de normalidad

Los test de normalidad sirven para evaluar si los datos siguen una distribución normal, esto es fundamental si quisieramos por ejemplo desarrollar algún modelo de inferencia ya que muchos de ellos suponen normalidad de los datos.

Existen diferentes test para probar la normalidad, algunos de ellos son:
- Shapiro-Wilk no es confiable para dataframes muy grandes.
- Kolmogorov-Smirnov: Es más adecuado para datos muy grandes, compara la distribuciòón empírica con la teórica.
- Anderson-Darling: es una extensión de KS pero sirve para detectar datos que tienen desviaciones en las colas.
"""

# Aplicar los tests a cada conjunto de datos
df_sales_no_num = df_sales[num_variables]
resultados = {}
for columna in df_sales_no_num.columns:
    resultados[columna] = test_normalidad(df_sales_no_num[columna], columna)
    print(resultados[columna])
    print("\n" + "="*80 + "\n")

"""Incluso después de eliminar los outliers, ambas variables (quantity y price) siguen sin tener una distribución normal, como lo indican los valores de los estadísticos y el p valor, al ser menor a 0.05 rechazamos la hipótesis nula. En caso de querer realizar una inferencia se podrían realizar transformaciones en los datos para lograr normalidad pero en este caso solo estamos realizando un análisis exploratorio.

## Análisis Bivariado

Con el objetivo de profundizar en los patrones temporales del comportamiento de compra, se realizó un análisis bivariado entre las variables mes y día de la semana. Con el fin de identificar cómo varía la actividad comercial a lo largo del tiempo, no solo en términos mensuales, sino también considerando la distribución de las transacciones dentro de cada semana.

Para ello, se utilizaron tablas de contingencia, tanto en frecuencia absoluta como relativa (%), y se aplicó una prueba de Chi-cuadrado, que permite evaluar si existe una relación estadísticamente significativa entre ambas variables. Además, se calculó la V de Cramér para cuantificar la intensidad de dicha relación.
"""

df_sales_cat = df_sales.select_dtypes(include=['object', 'category'])
df_sales_cat.head()

analisis_bivariado(df_sales_cat, 'mes', 'dia_de_semana')

"""El valor de V de Cramér (0.072) indica que existe una relación muy débil entre el mes y el día de la semana, aunque hay diferencias en la distribución de ventas por día, estas no varían fuertemente de un mes a otro. Sin embargo, los gráficos muestran que los martes y miércoles presentan una frecuencia de compra consistentemente mayor en casi todos los meses analizados. Por otro lado, los domingos son los días con menor volumen de transacciones, lo cual puede representar una oportunidad comercial: por ejemplo, implementar descuentos o promociones puntuales para incentivar las compras durante ese día. Este tipo de estrategia también podría extenderse a los lunes, que si bien no son los de menor actividad, no se encuentran entre los días más fuertes.

El análisis confirma que las ventas no se distribuyen de forma uniforme entre los días de la semana y que existen algunos picos estacionales definidos, especialmente en noviembre y diciembre. A pesar de esto, la relación entre los meses y los días es débil, lo que refuerza la idea de que el patrón semanal se mantiene relativamente constante independientemente del momento del año.

Dadas estas observaciones, la tienda podría beneficiarse al implementar promociones específicas en días de menor afluencia (como domingos y lunes) para estimular la demanda y equilibrar el flujo de ventas. En paralelo, dado que los martes y miércoles ya son días fuertes, se podría apuntar a incrementar el ticket promedio durante esos días mediante estrategias como bonificaciones por volumen de compra o productos complementarios en promoción.

## Análisis de series de tiempo

Dado que contamos con un dataset de ventas diarias, el análisis de series de tiempo es fundamental en nuestro caso. Este nos permite detectar e identificar tendencias a largo plazo y entender cómo evolucionan. Más allá que con los gráficos de evolución mensual y diaria ya habíamos detectado algunas tendencias, este análisis nos va a permitir entender estos patrones aún más.

Descomponer la serie nos permite entender tres conceptos fundamentales: **la tendencia, la estacionalidad y el ruido**. La tendencia refleja más que nada comportamientos a largo plazo, nosotros ya habíamos detectado en las gráficas un aumento de las ventas en los útlimos meses por lo que deberíamos detectarlo acá también. La estacionalidad se refiere más a patrones repetitivos y previsibles en períodos específicos, por ejemplo un aumento en las compras todos los viernes o en cierto mes. Como contamos con datos de solo un año se va a dificultar detectar tendencias mensuales pero si podemos hacerlo de forma diaria. El ruido o componente residual explica las variaciones aleatorias que no siguen ningún partrón específico.

Aunque este tipo de análisis se realiza mayoritariamente cuando se quieren ajustar modelos perdictivos y nosotros solo nos quedaremos con el análisis exploratorio, creemos que detectar estas tendencias nos puede llegar a servir por ejemplo, para detectar cuándo es mejor realizar una campaña de marketing o descubir si el aumento de ventas es sostenido en el tiempo.

Para llevar a cabo el análisis de series de tiempo, es necesario aplicar ciertas transformaciones sobre las variables temporales, con el fin de facilitar su manipulación y permitir un procesamiento adecuado de las fechas.
"""

df_sales_date = df_sales.copy()
df_sales_date.date = pd.to_datetime(df_sales_date.date).dt.date

transacciones_diarias = df_sales_date.groupby(df_sales_date['date']).agg({'bill_no': 'nunique'}).reset_index()
transacciones_diarias.rename(columns={'bill_no': 'cantidad_transacciones'}, inplace=True)
transacciones_diarias

# Convertir la columna 'fecha' al tipo datetime
transacciones_diarias['date'] = pd.to_datetime(transacciones_diarias['date'])
transacciones_diarias.info()

df=transacciones_diarias.copy()
df.set_index('date', inplace=True)
df.info()

# Configurar el estilo de las gráficas
# plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

cantidad_transacciones_txt='cantidad_transacciones'
# Visualizar la serie temporal completa
plt.figure(figsize=(10, 5))
plt.plot(df.index, df[cantidad_transacciones_txt], 'b-', linewidth=1.5)
plt.title('Cantidad de transacciones diarias (2010-2011)', fontsize=16)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Cantidad Transacciones', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nEstadísticas por mes (estacionalidad):")
monthly_stats = df.groupby([df.index.year, df.index.month])[cantidad_transacciones_txt].agg(['mean', 'std', 'min', 'max', 'sum']).round()
print(monthly_stats)

# Visualizar estadísticas mensuales (estacionalidad)
plt.figure(figsize=(10, 5))
monthly_stats['mean'].plot(kind='bar', yerr=monthly_stats['std'], capsize=4, color='skyblue')
plt.title('Valor Promedio Transacción por Mes (Todos los Años)', fontsize=16)
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Valor Promedio', fontsize=14)
plt.xticks(np.arange(12), ['Dic', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

"""Más allá de los que observamos en la gráfica, las estadísticas mensuales nos muestran una clara variación promedio de transacciones, esto indica la presencia de cierta estacionalidad, ya que los valores sugieren que ciertos meses tienen una mayor actividad. Esto podría estar asociado a ciertos eventos puentuales, campañas, vacaciones o meses de consumo particulares.

Si miramos la variación estandar, en diciembre del 2010 es cuando hay mayores fluctuaciones, si bien no es el mes con mayor ventas si es el que tiene mayores variaciones.

Observando el máximo y la suma de transacciones vemos que en que los meses que hay una venta total mayor, dic-2010 y nov-2011, también se encuentran los días con mayores ventas. Esto sugiere que el incremento en las ventas durante esos meses no se debe únicamente a un evento puntual o a un único día de alto volumen, sino que refleja un nivel de actividad elevado mantenido a lo largo del mes. Cabe destacar que los aumentos se observan cercanos a fin de año que son meses en los que se tiende a realizar mayores compras y en enero se observa una baja lo cual es esperable.

### Estacionariedad

A continuación se analiza la serie de tiempo correspondiente a la cantidad de transacciones diarias, con el objetivo de evaluar su comportamiento a lo largo del tiempo y determinar si presenta propiedades de estacionariedad, así como identificar tendencias o patrones relevantes para la toma de decisiones.
"""

# Analizar estacionariedad de la serie original
test_stationarity(df['cantidad_transacciones'], title='Cantidad de transacciones')

"""Tal como se anticipaba, los resultados de los test de estacionariedad confirman que la serie no es estacionaria. El test de Dickey-Fuller Aumentado (ADF) arroja un p-valor alto, por lo que no se rechaza la hipótesis nula de no estacionariedad. Por su parte, el test KPSS respalda esta conclusión desde un enfoque complementario: con un p-valor inferior a 0.05, se rechaza la hipótesis nula de estacionariedad.

La línea de media móvil suaviza notablemente la serie original, revelando inicialmente una caída en las transacciones, que posteriormente se revierte hacia el final del período, con un incremento significativo. Este comportamiento es coherente con lo observado en los gráficos presentados en la sección de visualizaciones.

Asimismo, se observa que al inicio del período la desviación estándar es elevada, lo cual indica alta volatilidad en las transacciones diarias. Con el paso del tiempo, esta variabilidad disminuye y se mantiene relativamente estable.

Detectamos que noviembre y diciembre son meses de alta actividad sostenida, lo que representa una excelente oportunidad para maximizar ventas si se planifican correctamente. Además, enero muestra una caída esperable en la demanda, lo que puede aprovecharse para estrategias específicas. La evolución de la variabilidad también sugiere un negocio más predecible hacia el final del período, lo que facilita la toma de decisiones operativas.

###  Autocorrelación - ACF y PACF

Para profundizar en la estructura temporal de la serie, se analizan las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF), con el objetivo de identificar la presencia de dependencias entre rezagos y orientar la selección de un modelo ARIMA adecuado.
"""

# Calcular y mostrar la función de autocorrelación (ACF)
plt.figure(figsize=(10, 4))
plot_acf(df['cantidad_transacciones'], lags=36, alpha=0.05, title='Función de Autocorrelación (ACF)')
plt.grid(True, alpha=0.3)

# Calcular y mostrar la función de autocorrelación parcial (PACF)
plt.figure(figsize=(10, 5))
plot_pacf(df['cantidad_transacciones'], lags=36, alpha=0.05, method='ywm', title='Función de Autocorrelación Parcial (PACF)')
plt.grid(True, alpha=0.3)
#plt.savefig('pacf.png', dpi=100)

"""En el gráfico de ACF se observa una lenta disminución de las autocorrelaciones a medida que aumentan los rezagos, con varios valores significativamente distintos de cero incluso después de 14 lags. Este patrón es típico de una serie no estacionaria. Además puede llegar a indicar que la serie tiene una memoria larga, los valores pasados siguen influyendo durante muchos períodos futuros, no solo depende de los 2 o 3 últimos días sino de más de 10 lo cual es de esperar.

En el PACF, el primer rezago tiene una autocorrelación parcial significativa y el segundo también muestra un valor claramente fuera del intervalo de confianza. Recién partir del tercer rezago, las barras caen dentro del intervalo de confianza, lo que sugiere un componente autoregresivo de orden 2, es decir, un AR(2).

Estas dos cosas nos indican que el modelo requiere diferenciación para alcanzar la estacionalidad, podría ser un ARIMA(2,1,0).

*  2 lags autoregresivos (AR)
*  1 diferenciación (para estacionarizar la serie)
*  0 lags de media móvil (MA)

En base a este análisis, se puede decir que, los días con alto o bajo volumen tienen un impacto prolongado durante varios días. Las ventas no se mantienen estables en el tiempo, sino que varían con patrones que se pueden modelar. Se puede construir un modelo que permita anticipar cómo se moverán las ventas en los próximos días. Con un año podemos trabajar una primera versión del modelo, sobre todo para entender patrones semanales y tendencias generales. Sin embargo, para capturar mejor la estacionalidad anual y construir un modelo más robusto, sería ideal contar con al menos 2 a 3 años completos de datos.

### Descomposición de la serie
"""

# Descomposición aditiva usando medias móviles
decomposition = seasonal_decompose(df['cantidad_transacciones'], model='additive', period=12)

# Visualizar la descomposición
fig = plt.figure(figsize=(8, 6))
plt.subplot(411)
plt.plot(decomposition.observed, 'k-', linewidth=1.5)
plt.title('Original', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(412)
plt.plot(decomposition.trend, 'b-', linewidth=1.5)
plt.title('Tendencia', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(413)
plt.plot(decomposition.seasonal, 'g-', linewidth=1.5)
plt.title('Estacionalidad', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(414)
plt.plot(decomposition.resid, 'r-', linewidth=1.5)
plt.title('Residuo (Ciclo + Ruido)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('descomposicion.png', dpi=100)

"""En este gráfico se muestra una descomposición aditiva de la serie, no sabemos si estamos frente a una serie aditiva o multiplicativa aún.

La linea de la tendencia, tal como habíamos visto anteriormente nos muestra una leve tendencia, suavizada, ascendente hacia el final del 2011 además de la caida en los primeros meses.

En el componente estacional se puede observar claramente un patronperiódico bien definido, con ciclos regulares de corta duración, senanal o quincenal, de acuerdo también al PACF.

En cuanto a los residuos vemos que están centrados en cero con una variabilidad más estable sobre todo en la segunda mitad del año, esto es bueno porque nos dice que la serie se explica en mayor medida por la tendencia y la estacionalidad, confirmando que fue bueno hacer esta descomposición.


En el primer mes de la serie, se observa un comportamiento anómalo reflejado en una variabilidad mucho mayor que en el resto del año. Esto se ve claramente tanto en la serie original como en el componente de residuos, donde hay picos pronunciados que no se repiten posteriormente. Este patrón sugiere que en ese periodo pudo haber ocurrido un evento extraordinario que distorsionó temporalmente el comportamiento normal de las transacciones.

Tanto el patrón que observamos en la grafica de ACF con 14 lags como la de estacionalidad confirman que hay ciclos tanto semanales como cada dos semanas, el ciclo semanal se da más que nada por los sábados que no se registran ventas posiblemente porque la tienda esté cerrada.

### Transformación en busca de estacionariedad

Resultados de los tests de estacionaridad sobre el dataframe sin outliers:
"""

test_stationarity(df['cantidad_transacciones'], title='Cantidad de transacciones', should_graph=False)

"""Dado que la serie original no cumple con los criterios de estacionariedad, se procede a aplicar una diferenciación de primer orden. Esta transformación consiste en restar cada valor de la serie con respecto a su valor anterior, lo que permite eliminar la tendencia y estabilizar la media a lo largo del tiempo.
Una vez obtenida la serie transformada, se vuelve a evaluar su estacionariedad mediante los tests de ADF y KPSS, con el objetivo de verificar si esta nueva versión de la serie está en condiciones de ser utilizada en modelos como ARIMA.
"""

# 1. Primera diferencia
df['primera_diferencia'] = df['cantidad_transacciones'].diff()
test_stationarity(df['primera_diferencia'].dropna(), title='Primera Diferencia')

"""Al aplicar la primera transformación, la media movil se estabiliza alrededoer de cero y la desviación estandar se mantiene constante. Además, ambos tests ADF y KPSS coinciden en indicar que la serie se ha vuelto estacionaria. Esta concordancia entre ambos tests refuerza la conclusión de que la primera diferenciación ha sido suficiente para estabilizar la media de la serie, haciéndola apta para modelado con técnicas como ARIMA, si quisieramos por ejemplo relizar predicciones. No es necesario realizar la transformación logaritmica ni una segunda diferenciación.

## Análisis de correlación
"""

df_sales_date.columns

"""#### Varianzas y correlación

Como paso preliminar al estudio de relaciones entre variables numéricas, se presenta la matriz de covarianza entre las dos principales variables cuantitativas del conjunto de datos: price (precio unitario del producto) y quantity (cantidad comprada por transacción). Esta matriz permite evaluar tanto la variabilidad individual de cada variable (a través de su varianza), como su relación conjunta (covarianza).
"""

varianza_correlacion(df_sales_date, "price", "quantity")

"""La varianza de price luego de la eliminación de los outliers es moderada, lo que indica una dispersión moderada en los precios respecto a su media. La varianza de quantity es significativamente mayor, reflejando una mayor variabilidad en las cantidades compradas por los clientes. La covarianza entre price y quantity es negativa (aunque debil), lo que sugiere una relación inversa: a medida que los precios aumentan, las cantidades tienden a disminuir, lo cual es de esperar.

Para corroborar dicha relación realizamos los siguientes calculos:
"""

# Calcular el coeficiente de correlación de Pearson
correlacion = df_sales_date['price'].corr(df_sales_date['quantity'], method='pearson')

print(f"Coeficiente de correlación de Pearson entre price y quantity: {correlacion:.4f}")

# Calcular coeficiente de Spearman
coef_spearman, p_value = spearmanr(df_sales_date['price'], df_sales_date['quantity'])

print(f"Coeficiente de correlación de Spearman: {coef_spearman:.4f}")
print(f"Valor p: {p_value:.4f}")

# --- MAPA DE CALOR DE CORRELACIÓN ---
print("\n=== Matriz de correlación (Pearson) ===")
corr_matrix = df_sales_date[['price', 'quantity']].corr(method='spearman')
print(corr_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="crest", fmt=".2f")
plt.title("Matriz de correlación- Price vs Quantity")
plt.show()

"""A partir del análisis realizado, se concluye que existe una relación negativa entre el precio (price) y la cantidad comprada (quantity). El coeficiente de correlación de Pearson es de -0.1706, lo que indica una correlación lineal negativa débil: a mayor precio, la cantidad comprada tiende a disminuir, aunque el efecto no es fuerte.

Sin embargo, al aplicar el coeficiente de correlación de Spearman, que es más robusto al evaluar relaciones monótonas (no necesariamente lineales), se obtiene un valor de -0.4077, con un valor p = 0.0000, indicando una relación negativa moderada y estadísticamente significativa.

Esto sugiere que, si bien la relación lineal directa es débil, sí existe una tendencia más clara (aunque no perfectamente lineal) a que cantidades mayores se asocien a precios más bajos. En términos prácticos para el cliente, esto puede indicar que los compradores tienden a adquirir más unidades cuando los precios son más accesibles, y menos cuando los precios suben, lo cual es consistente con un comportamiento racional del consumidor.

## Análisis de correspondencia

El análisis de correspondencia nos permite entender la correlación entre dos variables categóricas, en nuestro caso elegimos analizar la relación entre momento del día y día de semana para tratar de entender si por ejemplo, los viernes se compra más por la noche.
"""

df_sales_date.head()

# Tabla Pivot momento_del_dia y dia_de_semana
df_corresp = df_sales_date.pivot_table(index='momento_del_dia', columns='dia_de_semana', values='bill_no', aggfunc='count').fillna(0)
df_corresp

# Se crea el modelo para CA
ca = prince.CA(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=42)

ca = ca.fit(df_corresp)

ax = ca.plot_coordinates(
    X=df_corresp,
    ax=None,
    figsize=(10, 6),
    x_component=0,
    y_component=1,
    show_row_labels=True,
    show_col_labels=True)

print(f"ca.eigenvalues_: {ca.eigenvalues_}")
print(f"ca.total_inertia_: {ca.total_inertia_}")
print(f"ca.explained_inertia_: {ca.explained_inertia_}")

"""El análisis de correspondencias aplicado a las variables momento_del_día y día_de_semana indica que la relación entre ambas dimensiones es débil.La baja inercia total (0.0636) sugiere que la asociación global entre estas variables es limitada.

Visualmente, se identifican algunos patrones puntuales, como la proximidad entre “Mañana” y “Viernes”, o la distancia marcada de “Noche” respecto al resto, lo que puede reflejar comportamientos distintos en ciertos segmentos. Sin embargo, la mayoría de los puntos se agrupan cerca del centro, lo que evidencia una distribución bastante uniforme entre los días de la semana respecto a los momentos del día.

Por tanto, no se puede concluir que exista una relación fuerte o consistente entre momento_del_día y día_de_semana que justifique decisiones comerciales específicas basadas únicamente en esta combinación.

Vamos a válidar la relación entre estas varibales aplicando estadisticos.
"""

var1='momento_del_dia'
var2='dia_de_semana'
analisis_bivariado(df_sales_date, var1, var2, should_graph=False)

"""Los resultados del test Chi-cuadrado indican que existe una relación estadísticamente significativa entre el día de la semana y el momento del día (p < 0.05). Sin embargo, la magnitud de esta relación, evaluada mediante el coeficiente de V de Cramér (0.178), sugiere que dicha asociación es débil. Esto implica que, si bien hay ciertas combinaciones más frecuentes que otras, no se observan patrones marcadamente diferenciados entre los momentos del día y los días de la semana que justifiquen decisiones estratégicas basadas únicamente en esta relación.

Continuamos analizando otras relaciones
"""

top_cat = ventas_categorias.head(6).categoria.unique()
top_cat

df_sales_cat_top = df_sales_date.loc[df_sales_date.categoria.isin(top_cat)]
df_sales_cat_top

# Tabla cruzada: momento del mes vs categoría
tabla = pd.crosstab(df_sales_cat_top['dia_de_semana'], df_sales_cat_top['categoria'])
df_transacciones = df_sales_cat_top[['bill_no', 'categoria', 'dia_de_semana']].drop_duplicates()

# Crear modelo de correspondencia
ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42)

# Ajustar el modelo a la tabla
ca = ca.fit(tabla)

# Visualización de los ejes
ax = ca.plot_coordinates(
    X=tabla,
    ax=None,
    figsize=(6, 6),
    x_component=0,
    y_component=1,
    show_row_labels=True,
    show_col_labels=True)

ax.set_title("Análisis de Correspondencia: Momento del mes vs Categoría")

print(f"ca.eigenvalues_: {ca.eigenvalues_}")
print(f"ca.total_inertia_: {ca.total_inertia_}")
print(f"ca.explained_inertia_: {ca.explained_inertia_}")

var1='categoria'
var2='dia_de_semana'
analisis_bivariado(df_transacciones, var1, var2, should_graph=False)

"""A partir del análisis de correspondencia y la prueba estadística Chi-cuadrado, no se encuentra evidencia suficiente para afirmar que existe una relación significativa entre la categoría del producto y el día de la semana en que se realiza la compra. Esto se respalda con un valor p de 0.2619 (mayor al umbral común de 0.05), lo que implica que no se rechaza la hipótesis nula de independencia entre ambas variables. Además, el valor de V de Cramér (0.0091) refuerza esta conclusión, indicando que cualquier asociación presente es extremadamente débil.

Visualmente, el gráfico de correspondencias también muestra que las categorías y los días de la semana no presentan agrupaciones claras o asociaciones destacables. Por lo tanto, se concluye que el comportamiento de compra por día de la semana no varía sustancialmente entre las distintas categorías de productos analizadas.

**Analisis MCA**
"""

df=df_sales_cat_top.copy()

# Seleccionar columnas categóricas
df_mca = df[['momento_del_mes', 'momento_del_dia', 'categoria']].copy().astype('category')

# Análisis 1: momento del mes, momento del día y categoría
resultados_1 = ejecutar_mca(df_mca, titulo='Mapa de Categorías del MCA')

print("\n")

# Seleccionar columnas categóricas
df_mca = df[['momento_del_mes', 'dia_de_semana', 'categoria']].copy().astype('category')

# Análisis 2: momento del mes, día de semana y categoría
resultados_2 = ejecutar_mca(df_mca, titulo='Mapa de Categorías del MCA (con día de semana)')

print("---- Resultados 1 ----")
print("Valores propios:", resultados_1['eigenvalues'])  # Mide la varianza de cada componente
print("Varianza explicada:", resultados_1['explained_inertia'])  # Porcentaje de varianza explicada por cada eje
print("Coordenadas de las categorías:\n", resultados_1['coord_categorias'].head())
print("Coordenadas de los individuos (primeras filas):\n", resultados_1['coord_individuos'].head())

print("\n---- Resultados 2 ----")
print("Valores propios:", resultados_2['eigenvalues'])
print("Varianza explicada:", resultados_2['explained_inertia'])
print("Coordenadas de las categorías:\n", resultados_2['coord_categorias'].head())
print("Coordenadas de los individuos (primeras filas):\n", resultados_2['coord_individuos'].head())

print("---- Resultados 1 ----\n")
analisis_bivariado(df, 'momento_del_mes', 'categoria', should_graph=False)
print("\n---- Resultados 2 ----\n")
analisis_bivariado(df, 'momento_del_mes', 'dia_de_semana', should_graph=False)
print("\n---- Resultados 3 ----\n")
analisis_bivariado(df, 'momento_del_mes', 'momento_del_dia', should_graph=False)
print("\n---- Resultados 4 ----\n")
analisis_bivariado(df, 'categoria', 'dia_de_semana', should_graph=False)
print("\n---- Resultados 5 ----\n")
analisis_bivariado(df, 'categoria', 'momento_del_dia', should_graph=False)

"""A partir del análisis de correspondencias múltiples (MCA) y los cruces bivariados realizados, se observa que si bien existen relaciones estadísticamente significativas entre las variables temporales (momento del mes, día de la semana, momento del día) y las categorías de productos, la magnitud de dichas relaciones es muy débil. Esto se refleja en los valores de V de Cramér, que en todos los casos se mantienen por debajo de 0.04. Esto indica que las diferencias observadas en las distribuciones no son lo suficientemente marcadas como para considerar que ciertas categorías tienen un comportamiento de compra claramente dependiente del tiempo.

En términos prácticos, no hay evidencia suficiente para sostener que ciertas categorías de productos se vendan significativamente más en determinados momentos del mes, días de la semana o momentos del día. Aunque se detectaron diferencias estadísticamente significativas, son tan pequeñas que no justifican, por ejemplo, campañas de marketing o promociones segmentadas estrictamente según la variable temporal. Los gráficos de MCA refuerzan esta conclusión, ya que muestran una alta concentración de categorías y momentos cerca del centro del plano, sin agrupamientos claros. Asimismo, la baja inercia explicada por los primeros componentes evidencia que la variabilidad entre las variables analizadas no permite una diferenciación relevante.

## Medidas de similitud

Las medidas de similitud y desimilitud son herramientas fundamentales en el análisis de datos que nos permiten cuantificar qué tan parecidos o diferentes son los objetos en nuestros datasets. En nuestro caso queremos saber si existe similitud entre las categrorías creadas ya que puede ser un input para recomendar compras a los clientes. Por ejemplo si suelen comprar artículos de papelería podemos recomendarles cosas de decoración?

Para poder hacerlo, dado que tenemos una cantidad muy grande de productos, vamos a tomar el top de ventas de cada categoría.
"""

print("Cantidad de clientes únicos:", df['customer_id'].nunique())

print(f'Cantidad de items {df_sales_date.item_name.nunique()} distintos de categorías {df_sales_date.categoria.nunique()}')

# Crear matriz binaria: 1 si el cliente compró al menos una vez en esa categoría
df_binaria = df_sales_date[['customer_id', 'categoria']].drop_duplicates()
df_binaria = df_binaria[df_binaria['customer_id'] != 99999]
df_binaria.head()

matriz_binaria = pd.crosstab(df_binaria['categoria'], df_binaria['customer_id'])
matriz_binaria.head()

# Matriz de características (sin la columna customer_id)
features = matriz_binaria.values
features

# Obtener el índice correspondiente a las características (que es el índice de matriz_binaria)
categorias = matriz_binaria.index.values
categorias

# Calcular distancias
manhattan_dist = manhattan_distances(features)
euclidean_dist = euclidean_distances(features)

# Convertir distancias a similitudes
manhattan_sim = 1 / (1 + manhattan_dist)
euclidean_sim = 1 / (1 + euclidean_dist)

"""Se comentan los prints para mejorar la lectura"""

manhattan_df = pd.DataFrame(manhattan_sim, index=categorias, columns=categorias)
# print("\nSimilitud basada en Distancia Manhattan:")
# print(manhattan_df.round(3))

euclidean_df = pd.DataFrame(euclidean_sim, index=categorias, columns=categorias)
# print("\nSimilitud basada en Distancia Euclidiana:")
# print(euclidean_df.round(3))

# Calcular similitud Jaccard entre todos los pares de customers
n_users = len(features)
jaccard_sim = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i == j:
            jaccard_sim[i, j] = 1.0
        else:
            jaccard_sim[i, j] = jaccard_score(features[i], features[j])

jaccard_df = pd.DataFrame(jaccard_sim, index=categorias, columns=categorias)
jaccard_df.head()

# Llamamos para todos los ejemplos de la muestra comparacion_metodos_similitud
for i in range(0, len(categorias)):
    comparacion_metodos_similitud(categorias[i], manhattan_df, euclidean_df, jaccard_df)

print("\nObservaciones:")
print("- Diferentes métodos pueden dar diferentes recomendaciones")

"""Para poder visualizar, realizamos las graficas sobre las matrices de similitud."""

# Visualización de las matrices de similitud
fig, (ax1) = plt.subplots(1, figsize=(18, 6))

# Similitud Manhattan
im1 = ax1.imshow(manhattan_sim, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax1.set_title('Similitud Manhattan')
ax1.set_xticks(range(len(categorias)))
ax1.set_yticks(range(len(categorias)))
ax1.set_xticklabels(categorias, rotation=45)
ax1.set_yticklabels(categorias)
plt.colorbar(im1, ax=ax1)

plt.tight_layout()
plt.show()

# Visualización de las matrices de similitud
fig, (ax2) = plt.subplots(1, figsize=(18, 6))

# Similitud euclidiana
im2 = ax2.imshow(euclidean_sim, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax2.set_title('Similitud Euclidiana')
ax2.set_xticks(range(len(categorias)))
ax2.set_yticks(range(len(categorias)))
ax2.set_xticklabels(categorias, rotation=45)
ax2.set_yticklabels(categorias)
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

# Visualización de las matrices de similitud
fig, (ax3) = plt.subplots(1, figsize=(18, 6))

# Similitud Jaccard
im3 = ax3.imshow(jaccard_sim, cmap='Greens', aspect='auto', vmin=0, vmax=1)
ax3.set_title('Similitud Jaccard')
ax3.set_xticks(range(len(categorias)))
ax3.set_yticks(range(len(categorias)))
ax3.set_xticklabels(categorias, rotation=45)
ax3.set_yticklabels(categorias)
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()

"""Como estas visualizaciones no son tan representativas, buscamos otro tipo de visualización"""

import networkx as nx

G = nx.Graph()

for category in categorias:
    G.add_node(category)

similarity_threshold = 0.6
for i in range(len(categorias)):
    for j in range(i + 1, len(categorias)):
        if jaccard_sim[i, j] > similarity_threshold:
            G.add_edge(categorias[i], categorias[j], weight=jaccard_sim[i, j])

nx.draw(G, with_labels=True)
plt.show()

"""Para mayor visualizacion eliminamos las categorias que tienen una similitud de menos de 0.6"""

# Crear matriz binaria: 1 si el cliente compró al menos una vez en esa categoría
df_binaria = df_sales_date[['customer_id', 'categoria']].drop_duplicates()
df_binaria = df_binaria[df_binaria['customer_id'] != 99999]
df_binaria = df_binaria[~df_binaria['categoria'].isin(['Limpieza', 'Muebles', 'Carnes', 'Cintas y Manualidades', 'Tecnologia u Oficina', 'Salud y Ciuidado Personal', 'Cocina Vintage', 'Velas y Aromas', 'Repostería y Accesorios', 'Infantil / Juguetes', 'Relojes y Despertadores', 'Materiales Naturales', 'Decoración de Hogar', 'Velas y Aromas', 'Panificados', 'Salud y Cuidado Personal', 'Ropa, Textiles y accesorios', 'Tecnología u Oficina', 'Accesorios del Hogar', 'Dulces', 'Frutas', 'Lácteos', 'Jardinería y Exterior', 'Bebidas', 'Navidad y Festividades'])]

matriz_binaria = pd.crosstab(df_binaria['categoria'], df_binaria['customer_id'])

# Matriz de características (sin la columna customer_id)
features = matriz_binaria.values
categorias = matriz_binaria.index.values
categorias

# Calcular similitud Jaccard entre todos los pares de customers
n_users = len(features)
jaccard_sim = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i == j:
            jaccard_sim[i, j] = 1.0
        else:
            jaccard_sim[i, j] = jaccard_score(features[i], features[j])

jaccard_df = pd.DataFrame(jaccard_sim, index=categorias, columns=categorias)

G = nx.Graph()

for category in categorias:
    G.add_node(category)

similarity_threshold = 0.6
for i in range(len(categorias)):
    for j in range(i + 1, len(categorias)):
        if jaccard_sim[i, j] > similarity_threshold:
            G.add_edge(categorias[i], categorias[j], weight=jaccard_sim[i, j])

nx.draw(G, style='dotted', with_labels=True)
plt.show()

"""Con esta gráfica logramos obtener los que tienen una relación más fuerte.

Se aplicaron tres métodos para calcular la similitud entre categorías de productos: distancia de Manhattan, distancia Euclidiana y el índice de similitud de Jaccard. El objetivo fue explorar la posibilidad de generar recomendaciones cruzadas entre categorías, partiendo de la hipótesis de que ciertos grupos de productos tienden a comprarse en conjunto. Esta información es útil para diseñar campañas de marketing personalizadas, mejorar estrategias de recomendación y optimizar la gestión del surtido.

Los resultados confirmaron la utilidad del índice de Jaccard frente a los métodos de distancia tradicionales (Manhattan y Euclidiana). Mientras estos últimos arrojaron valores de similitud muy bajos y con poca variabilidad (en torno a 0.001–0.05), Jaccard fue capaz de captar diferencias relevantes entre categorías. Por ejemplo, se observaron similitudes destacadas entre:

“Papelería y Regalos” y “Bolsas y Organizadores” (0.650),

“Ornamentos y Figuras” y “Decoración” (0.721),

“Papelería y Regalos” y “Otros” (0.830),
lo que sugiere que existen patrones de consumo comunes entre dichas categorías.

Asimismo, se reafirmó una hipótesis previa: la categoría “Lácteos”, que fue creada para agrupar objetos vinculados al tratamiento de productos lácteos (como frascos, jarros o utensilios), mostró una fuerte similitud con “Ornamentos y Figuras” y “Decoración de Pared”. Esto valida la idea de que clientes interesados en artículos decorativos también adquieren productos funcionales o temáticos que complementan un estilo de vida o decoración del hogar.

Las asociaciones identificadas permiten inferir posibles compras complementarias, como la relación entre “Panificados” y “Repostería”, o entre “Navidad y Festividades” y “Ornamentos”, lo cual es valioso para sugerir promociones cruzadas entre categorías fuertes y categorías menos rotativas. En efecto, este análisis puede ser base para implementar recomendaciones personalizadas y promociones de venta cruzada en la tienda o canal online.

## Analisis sobre categoría Otros

A partir del análisis realizado, se busca confirmar si la categoría “Otros”, que forma parte del top de categorías por volumen, contiene algún ítem relevante que merezca ser destacado. El objetivo es identificar si alguno de estos productos, actualmente clasificados como genéricos, podría representar una oportunidad específica o formar parte de una recomendación comercial más precisa.
"""

# Cargar el archivo CSV
df = df_sales_date.copy()

# Filtrar los ítems cuya categoría sea "otros"
df_otros = df[df['categoria'].str.lower() == 'otros']

# Agrupar por item_name y sumar la cantidad vendida
top_otros = df_otros.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(20)

# Crear el gráfico
plt.figure(figsize=(15, 6))

# Generar colores distintos usando la paleta 'tab20'
colors = cm.get_cmap('tab20', 20).colors

# Crear gráfico de barras con colores distintos
bars = plt.barh(top_otros.index[::-1], top_otros.values[::-1], color=colors)

# Etiquetas de valores al final de cada barra
for i, v in enumerate(top_otros.values[::-1]):
    plt.text(v + 100, i, f'{v:,}', va='center', fontsize=9)

# Etiquetas de ejes y título
plt.xlabel('Cantidad de Ventas')
plt.ylabel('Item')
plt.title('Cantidad de Ventas por Top 20 de Item (Categoría: Otros)')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()

"""Se observa que ninguno de los ítems clasificados dentro de la categoría "Otros" figura entre los productos más vendidos, por lo que se concluye que su presencia no impacta significativamente en el análisis principal ni en las conclusiones derivadas.

## Análisis de relación entre precio promedio y volumen de ventas por producto

Con el objetivo de identificar productos estratégicos tanto desde el punto de vista del valor unitario como del volumen de ventas, se realizó un análisis cruzado entre los productos con mayor precio promedio y aquellos con mayor cantidad total vendida. Asimismo, se examinó si los productos de menor precio también representan una parte significativa del volumen de ventas.
Este análisis permite detectar:

Productos premium de alta rotación, que combinan valor y demanda.

Productos económicos clave, que impulsan el volumen general de ventas.

Posibles oportunidades para focalizar campañas comerciales o ajustar márgenes según el comportamiento del consumidor.
"""

# Copiar el DataFrame
df = df_sales_date.copy()

# Asegurarse de que el campo 'price' sea numérico
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Agrupar por item_name:
# - Precio promedio
# - Total vendido (cantidad)
df_stats = df.groupby('item_name').agg(
    precio_promedio=('price', 'mean'),
    cantidad_vendida=('quantity', 'sum')
).reset_index()

# Ordenar por precio
top_caros = df_stats.sort_values(by='precio_promedio', ascending=False).head(20)

# Ordenar por cantidad vendida
top_vendidos = df_stats.sort_values(by='cantidad_vendida', ascending=False).head(20)

# Ver cuáles coinciden
coinciden = pd.merge(top_caros, top_vendidos, on='item_name')

# Mostrar solo los nombres de ítems que están tanto en el top 20 de los más caros como en el top 20 de los más vendidos
coinciden[['item_name']]

# Ítems más baratos (precio promedio más bajo)
top_baratos = df_stats.sort_values(by='precio_promedio', ascending=True).head(20)

# Ítems más vendidos (ya definido como top_vendidos)
# Buscar coincidencias entre los más baratos y más vendidos
coinciden_baratos = pd.merge(top_baratos, top_vendidos, on='item_name')

coinciden_baratos[['item_name']]

"""Los resultados indican que no existe coincidencia entre los ítems más caros y los menos vendidos, así como tampoco entre los más baratos y los de mayor volumen. Esto sugiere que el comportamiento del consumidor no está orientado únicamente por el precio, sino que probablemente intervienen otros factores como la utilidad, estacionalidad, percepción de valor o estrategia comercial.

Esta ausencia de coincidencias refuerza la idea de que el éxito de ventas no depende exclusivamente del posicionamiento en precio, y destaca la importancia de comprender el contexto específico de cada producto para optimizar su desempeño en el mercado.

# Etapa IV: Resultados y Discusión

**Interpretación de los resultados del análisis**

Los resultados del análisis visual y estadístico delineados en la Etapa III proporcionan una comprensión clara de la dinámica del negocio y responden directamente a la problemática planteada. En primer lugar, se confirmó que un grupo reducido de categorías y productos lidera las ventas. Categorías como Papelería y Regalos o Bolsas y Organizadores son pilares del ingreso, lo que indica que el catálogo tiene áreas de especial fortaleza.

Esta concentración sugiere que una proporción relativamente pequeña del inventario genera casi la mitad de las ventas.

A nivel de producto ocurre algo similar, con algunos “best-sellers” vendiendo decenas de miles de unidades mientras la mayoría de los ítems tienen volúmenes más modestos. Este patrón de ventas concentradas es común en el comercio minorista y tiene implicaciones importantes: por un lado, ofrece la oportunidad de especializarse y potenciar aquello que más demanda tiene; por otro, plantea el riesgo de dependencia excesiva en unos pocos productos, que podría ser problemático si las tendencias de consumo cambian o si surgen problemas de stock.

En segundo lugar, el análisis temporal reveló fuertes componentes estacionales. Durante gran parte del año 2011, las ventas mensuales se mantuvieron en un rango estable, pero el último trimestre vio un aumento exponencial en ingresos. Esta subida coincide con la temporada previa a las fiestas navideñas, lo que estadísticamente se manifiesta en una tendencia alcista significativa de septiembre a noviembre (las ventas de noviembre más que duplicaron a las de agosto).

Al indagar en las causas, determinamos que el factor decisivo fue la afluencia de más clientes comprando en ese período, más que cambios en el gasto medio de cada cliente. En efecto, encontramos una correlación entre el número de clientes únicos mensuales y el monto total vendido, mientras que la relación entre el ticket promedio y las ventas fue débil. Esto significa que la estrategia de captación o las condiciones de mercado en fin de año ampliaron la base de compradores, traduciéndose en más transacciones y mayor facturación global, sin que cada venta individualmente fuese mucho más grande. Desde un punto de vista de negocio, este hallazgo sugiere que el crecimiento logrado provino de la expansión del alcance de mercado (más clientes), posiblemente a través de promociones estacionales, marketing navideño, antes que de cambios en la conducta de compra de los clientes habituales.

En tercer lugar, el estudio de las métricas de comportamiento de los clientes indicó que la fidelidad promedio se mantuvo estable en torno a 1.5 compras por cliente en cada mes, lo cual es relativamente bajo. La mayoría de los clientes compró solo una vez por mes y no hubo indicios de un aumento general en la frecuencia de recompra en los meses pico.

Sin embargo, identificamos la existencia de clientes “heavy users” o grandes compradores que efectuaron numerosas, por ejemplo, los identificados con ID 12748 y 17841. Su aporte, si bien numéricamente es minoritario en términos absolutos, tuvo impacto en meses específicos elevando ligeramente el promedio de compras por cliente. Esto sugiere que podrían tratarse de clientes mayoristas o comerciales (por ejemplo, revendedores) que valdría la pena investigar su procedencia. Para el negocio, reconocer a estos actores es crucial, ya que pueden representar una especie de “cliente VIP” cuyo comportamiento no solo genera ingresos directos significativos sino que también puede distorsionar las métricas generales si no se analiza por separado. No observamos, que su presencia altere la tendencia macro: excluyendo a estos outliers, la conclusión central sigue siendo que el grueso de los clientes realiza compras únicas o poco frecuentes, y que el salto estacional de ventas vino de sumar muchos clientes nuevos más que de multiplicar las compras de los ya existentes.

En síntesis, desde un punto de vista global, la empresa logró aumentar sustancialmente sus ventas apoyándose en categorías de productos clave y en una exitosa captación de clientes en la temporada alta. No hubo cambios estructurales en el gasto promedio por cliente ni en la diversidad de productos por compra que explicaran el crecimiento, lo que implica que las estrategias de marketing o las circunstancias externas que atrajeron más compradores fueron el factor principal de éxito en el período analizado. Estas conclusiones, respaldadas tanto por las visualizaciones como por la cuantificación estadística básica (por ej., correlaciones, promedios mensuales), ofrecen una base para la toma de decisiones.

A continuación, se presentan recomendaciones accionables, con el objetivo de aprovechar las oportunidades detectadas y mitigar posibles riesgos para el negocio.

**Recomendaciones accionables para el cliente**

A la luz de los resultados obtenidos, se proponen las siguientes acciones concretas orientadas a potenciar las ventas y optimizar la gestión del negocio:

Focalizar la estrategia en categorías líderes

Dado que “Papelería y Regalos”, “Bolsas y Organizadores” y “Decoración” generan casi la mitad de las ventas, se recomienda priorizar estas categorías en las estrategias comerciales. Esto incluye asegurar stock suficiente de sus productos más vendidos, mejorar su exhibición (en tienda o en la web) y destinar presupuesto de marketing para promocionarlos. Al mismo tiempo, conviene profundizar en el análisis de estas categorías para comprender qué factor común las hace exitosas (¿son productos de regalo populares, tienen buen precio, alta demanda estacional?) y replicar ese éxito en otras categorías emergentes.

Gestión del portafolio de productos de baja rotación

Las categorías menos vendidas (que en conjunto contribuyen con una fracción de los ingresos) merecen una revisión. Se sugiere evaluar la rentabilidad y el rol estratégico de esos productos de baja rotación. En algunos casos, podría ser beneficioso racionalizar la oferta, discontinuando productos que sistemáticamente muestran ventas muy bajas, para reducir costos de almacenamiento y complejidad operativa. Alternativamente, si ciertas categorías pequeñas (por ej., “Salud y Cuidado Personal” o “Limpieza”) son importantes para mantener un surtido completo, se sugiere explorar acciones para impulsarlas: promociones cruzadas (combinar productos de esas categorías con los más vendidos), descuentos o paquetes que incrementen su visibilidad y atractivo. Por ejemplo, se podrían generar promociones cruzadas entre categorías líderes como “Decoración” y categorías de menor volumen como “Lácteos”, aprovechando los resultados del análisis de similitud, donde se identificó una relación bajo el índice de Jaccard. Del mismo modo, también se observaron similitudes relevantes entre “Papelería y Regalos” y “Bolsas y Organizadores”, lo que habilita la posibilidad de diseñar combinaciones estratégicas entre categorías con diferente rotación, impulsando así el movimiento de productos con menor volumen.

Fortalecer la planificación estacional

La marcada estacionalidad detectada aconseja una planificación proactiva de inventario y marketing de cara a los meses de alta demanda. Se recomienda incrementar inventarios de los productos más demandados antes de septiembre, para cumplir con el pico de ventas de fin de año sin quiebres de stock. Igualmente, conviene diseñar con antelación campañas de marketing estacional (p. ej., campañas navideñas) enfocadas en los artículos y categorías estrella identificados. Dado que muchas ventas adicionales provinieron de clientes nuevos en esos meses, sería útil lanzar promociones de captación temprana (por ejemplo, ofertas de “early Christmas shopping” en octubre) para adelantar parte de la demanda y distribuir mejor el volumen, evitando cuellos de botella logísticos en noviembre. Asimismo, después del pico, preparar acciones para enero (mes tradicionalmente lento) como liquidaciones o ventas especiales podría amortiguar la caída poste fiestas y fidelizar a los clientes adquiridos en diciembre.

Estrategias de adquisición y fidelización de clientes

El análisis mostró que el crecimiento vino de conseguir más clientes, mientras que la tasa de recompra por cliente permaneció baja. Por ello, se aconseja invertir en estrategias de adquisición de clientes durante todo el año, no solo en temporada alta. Esto puede implicar campañas digitales, alianzas comerciales u ofertas de primera compra para atraer nuevos compradores constantemente. Al mismo tiempo, existe una gran oportunidad para aumentar la frecuencia de compra de la base actual de clientes. Implementar programas de fidelización (por ejemplo, un sistema de puntos o descuentos progresivos por compras recurrentes) podría incentivar a los clientes a realizar más de una compra al año. Incluso un pequeño aumento en la frecuencia media (de 1.5 a 2 compras por cliente al año, por ejemplo) repercutiría positivamente en las ventas totales. Identificar a los clientes satisfechos de la temporada alta y enviarles ofertas personalizadas en meses intermedios puede convertir clientes estacionales en clientes regulares.

Atención a los clientes “VIP” o mayoristas

Los pocos clientes que realizan compras excepcionalmente frecuentes (identificados en el análisis) representan un segmento valioso. Es recomendable contactar directamente con estos clientes para entender sus necesidades y motivaciones. Si son minoristas comprando al por mayor, podría instaurarse un programa específico de ventas mayoristas, con precios por volumen u otras facilidades, que los anime a seguir eligiendo la empresa como proveedor principal. Si son simplemente clientes individuales muy fieles, un trato preferencial, como acceso anticipado a nuevos productos, envíos gratuitos, o un gestor de cuentas dedicado. En cualquier caso, monitorizar periódicamente el comportamiento de estos grandes compradores resulta útil, cualquier señal de disminución en sus compras podría indicar la necesidad de una intervención comercial.


Optimización de precios y promociones basada en datos

La variabilidad de precios entre categorías (y la presencia de outliers de precio) sugiere revisar la estrategia de precios. Productos en categorías como Muebles, con precios consistentemente altos, podrían requerir una estrategia de valor agregado para justificar su costo. Por otro lado, para las categorías con mucha competencia de productos similares, se podrían implementar promociones segmentadas. Por ejemplo, si Panificados tiene precios medianos altos pero compite con alternativas, ofrecer descuentos temporales podría aumentar su rotación manteniendo margen gracias al precio base elevado. En contraste, en categorías de artículos de bajo precio unitario pero alto volumen (como Papelería y Regalos), quizás convenga crear bundles o packs promocionales para elevar el ticket promedio por transacción. Todas estas tácticas podrian ser evaluadas con pequeñas pruebas A/B.

En conclusión, las recomendaciones anteriores buscan capitalizar las fortalezas reveladas por el análisis (productos y categorías más rentables, temporadas fuertes, clientes fieles) y abordar las debilidades o áreas de mejora (dependencia en pocos ítems, estacionalidad pronunciada, baja recurrencia promedio). Al implementar estas acciones, la empresa podrá no solo aumentar sus ventas de forma sostenible, sino también lograr un crecimiento equilibrado, diversificando riesgos y construyendo relaciones más sólidas con su base de clientes. Cada recomendación deberá monitorearse con métricas específicas (como crecimiento por categoría, tasa de repetición de compra, stock out rate en temporada alta, etc.) para afinar la estrategia sobre la marcha, fomentando así una cultura de decisiones basadas en datos en la organización.
"""