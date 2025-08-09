import glob
import pandas as pd

archivos = glob.glob("Data/mammography_spectra/Mo_unfiltered_10kV-50kV/*.dat")  # lista de archivos con búsqueda de la ruta
Mo = {}
for archivo in archivos:
    Mo_ind = pd.read_csv(archivo, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    Mo[archivo] = Mo_ind

archivos2 = glob.glob("Data/mammography_spectra/Rh_unfiltered_10kV-50kV/*.dat") 
Rh = {}
for archivo2 in archivos2:
    Rh_ind = pd.read_csv(archivo2, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    Rh[archivo2] = Rh_ind
    
    
archivos3 = glob.glob("Data/mammography_spectra/W_unfiltered_10kV-50kV/*.dat")
W = {}
for archivo3 in archivos3:
    W_ind = pd.read_csv(archivo3, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    W[archivo3] = W_ind