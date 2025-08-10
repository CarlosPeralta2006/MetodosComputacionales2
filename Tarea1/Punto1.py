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
 
# Gráfica

def plot_spectra(Mo, Rh, W):
    elementos = {"Molibdeno (Mo)": Mo, "Rodio (Rh)": Rh, "Tungsteno (W)": W}  # organizar info en un diccionario

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # plot con 3 subplots

    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # toma cada 2 voltajes, podría ser cada 3

        colores = plt.cm.plasma(np.linspace(0, 1, len(voltajes_filtrados)))  # estilo

        for color, kv in zip(colores, voltajes_filtrados):
            df = dicc[kv]  # DataFrame con columnas "Energía", "Fotones"
            ax.plot(df["Energía"], df["Fotones"], label=f"{kv}", color=color, linewidth=1.2)

        ax.set_title(nombre, fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Solo un label en y común
    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)

    # Leyenda global 
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10, frameon=False)
    plt.subplots_adjust(top=1.0, bottom=0.25, wspace=0.15)
    plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

plot_spectra(Mo, Rh, W)
