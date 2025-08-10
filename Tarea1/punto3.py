# Seleccionar el DataFrame de Mo a 30kV
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Obtención de Datos
def data():
    archivos = glob.glob("Data/mammography_spectra/Mo_unfiltered_10kV-50kV/*.dat")  # lista de archivos con búsqueda de la ruta

    Mo = {}
    for archivo in archivos:
        Mo_ind = pd.read_csv(archivo, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=14, encoding="latin1")
        arch = archivo[52:-4]
        Mo[arch] = Mo_ind

    archivos2 = glob.glob("Data/mammography_spectra/Rh_unfiltered_10kV-50kV/*.dat") 
    Rh = {}
    for archivo2 in archivos2:
        Rh_ind = pd.read_csv(archivo2, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=14, encoding="latin1")
        arch2 = archivo2[52:-4]
        Rh[arch2] = Rh_ind

    archivos3 = glob.glob("Data/mammography_spectra/W_unfiltered_10kV-50kV/*.dat")
    W = {}
    for archivo3 in archivos3:
        W_ind = pd.read_csv(archivo3, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=14, encoding="latin1")
        arch3 = archivo3[50:-4]
        W[arch3] = W_ind
    
    return Mo, Rh, W

spectra = data()
Mo, Rh, W = spectra

# Obtener la primera clave del diccionario Mo
primera_clave = next(iter(Mo.keys()))

# Obtener el DataFrame asociado a esa primera clave
primer_df_mo = Mo[primera_clave]