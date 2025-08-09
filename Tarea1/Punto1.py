import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Obtención de Datos

archivos = glob.glob("Data/mammography_spectra/Mo_unfiltered_10kV-50kV/*.dat")  # lista de archivos con búsqueda de la ruta

Mo = {}
for archivo in archivos:
    Mo_ind = pd.read_csv(archivo, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    arch = archivo[52:-4]
    Mo[arch] = Mo_ind

archivos2 = glob.glob("Data/mammography_spectra/Rh_unfiltered_10kV-50kV/*.dat") 
Rh = {}
for archivo2 in archivos2:
    Rh_ind = pd.read_csv(archivo2, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    arch2 = archivo2[52:-4]
    Rh[arch2] = Rh_ind

archivos3 = glob.glob("Data/mammography_spectra/W_unfiltered_10kV-50kV/*.dat")
W = {}
for archivo3 in archivos3:
    W_ind = pd.read_csv(archivo3, sep=r"\s+", header=None, names=["Energía", "Fotones"], skiprows=15, encoding="latin1")
    arch3 = archivo3[50:-4]
    W[arch3] = W_ind

# Gráfica

def plot_spectra(Mo, Rh, W, filename="1.pdf"):
    elementos = {"Mo": Mo, "Rh": Rh, "W": W}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # toma cada 2 voltajes para no saturar

        # Crear un mapa de colores según cantidad de voltajes que se mostrarán
        colores = plt.cm.viridis(np.linspace(0, 1, len(voltajes_filtrados)))

        for color, kv in zip(colores, voltajes_filtrados):
            df = dicc[kv]  # DataFrame con columnas ["Energía", "Fotones"]
            ax.plot(df["Energía"], df["Fotones"],
                    label=f"{kv} kV",
                    color=color,
                    linewidth=1.2)

        ax.set_title(nombre, fontsize=14)
        ax.set_xlabel("Energía (keV)")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Solo un ylabel común
    axes[0].set_ylabel("Conteo de fotones (u.a.)")

    # Leyenda global debajo de todos los subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

plot_spectra(Mo, Rh, W)
