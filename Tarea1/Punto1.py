import glob
import pandas as pd

archivos = glob.glob("Data/mammography_spectra/Mo_unfiltered_10kV-50kV/*.dat")  # lista de archivos con su ruta

for archivo in archivos:
    Mo = pd.read_csv(archivo, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
archivos2 = glob.glob("Data/mammography_spectra/Rh_unfiltered_10kV-50kV/*.dat") 

for archivo2 in archivos2:
    Rh = pd.read_csv(archivo2, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    
archivos3 = glob.glob("Data/mammography_spectra/W_unfiltered_10kV-50kV/*.dat")

for archivo3 in archivos3:
    W = pd.read_csv(archivo3, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")