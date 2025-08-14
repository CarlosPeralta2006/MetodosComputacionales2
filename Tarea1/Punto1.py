import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# Obtención de Datos
def data():
    archivos = glob.glob("Data/mammography_spectra/Mo_unfiltered_10kV-50kV/*.dat")  # lista de archivos con búsqueda de la ruta
    Mo = {}   # diccionario que contendrá la información: llaves será el kilovoltaje, info será un dataframe con la energía y el conteo de fotones
    for archivo in archivos:
        Mo_ind = pd.read_csv(archivo, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=14, encoding="latin1")  # archivo separado por espacios, 14 lineas saltadas y codificado en latin1
        # codificación latin1 evita errores de lectura, más amplio
        arch = archivo[52:-4]  # deja únicamente de llave "--kV" 
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
    
    return Mo, Rh, W  # retorna 3 diccionarios, cada unoi con un elemento diferente.

spectra = data()
Mo, Rh, W = spectra
 
# Gráfica

def plot_spectra(Mo, Rh, W):
    elementos = {"Molibdeno (Mo)": Mo, "Rodio (Rh)": Rh, "Tungsteno (W)": W}  # organizar info en un diccionario

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # plot con 3 subplots
    cmap = cm.plasma 
    for ax, (nombre, dicc) in zip(axes, elementos.items()):  # zip para facilitar la lectura de los datos, emparejándolos
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # toma cada 2 voltajes para no saturar y mantener tendencia
        
        # colores en gradiente para la colorbar en lugar de leyenda 
        kv_numericos = [float(kv.replace("kV", "")) for kv in voltajes_filtrados]
        norm = mcolors.Normalize(vmin=min(kv_numericos), vmax=max(kv_numericos))        

        for kv in voltajes_filtrados:
            kv_num = float(kv.replace("kV", ""))
            df = dicc[kv]  # DataFrame con columnas "Energía", "Fotones"
            ax.plot(df["Energía"], df["Fotones"],  color=cmap(norm(kv_num)), linewidth=1.2)

        ax.set_title(nombre, fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Solo un label en y común
    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)

    # Barra de gradiente 
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # requerido para colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.12)
    cbar.set_label("Voltaje del tubo (kV)", fontsize=12)
    
    plt.subplots_adjust(top=1.0, bottom=0.25, wspace=0.15)
    plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

plot_spectra(Mo, Rh, W)
