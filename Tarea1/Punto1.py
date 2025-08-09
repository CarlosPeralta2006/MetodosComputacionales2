import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner
import os
from google.colab import drive
import scipy.optimize as spo
import math

import pandas

import os
import pandas as pd
import numpy as np

# Configuración - CAMBIA ESTAS RUTAS
carpeta_origen = "Metodos II/mammography_spectra/Mo_unfiltered_10kV-50kV"
carpeta_destino = "Metodos II/mammography_spectra/CSV_convertidos"

# Crear carpeta de destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

def procesar_archivo_dat(ruta_entrada, ruta_salida):
    """Procesa un archivo .dat y lo guarda como CSV"""
    with open(ruta_entrada, 'r') as f:
        lineas = [linea.strip() for linea in f if not linea.startswith('#') and not linea.startswith('*')]
    
    # Extraer datos numéricos
    datos = []
    for linea in lineas:
        if linea and not linea.startswith(('---', '###')):
            valores = linea.split()
            if len(valores) >= 2:  # Asegura que tenga al menos energía y fluencia
                datos.append([float(valores[0]), float(valores[1])])
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(datos, columns=['Energia_keV', 'Fluencia'])
    df.to_csv(ruta_salida, index=False)

# Procesar TODOS los archivos .dat en la carpeta
for archivo in os.listdir(carpeta_origen):
    if archivo.endswith('.dat'):
        entrada = os.path.join(carpeta_origen, archivo)
        salida = os.path.join(carpeta_destino, archivo.replace('.dat', '.csv'))
        procesar_archivo_dat(entrada, salida)
        print(f"Convertido: {archivo}")

print("¡Conversión completada!")