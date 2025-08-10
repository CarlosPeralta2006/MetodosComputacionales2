from scipy.signal import find_peaks
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Punto1 import data

# Obtener los datos
spectra = data()
Mo, Rh, W = spectra

from scipy.signal import find_peaks

# Función para detectar los índices de los picos y sus zonas
def detectar_zona_picos(df, factor_height=0.15, factor_prom=0.08, factor_bajo=0.35):
    x = df['Energía'].values
    y = df['Fotones'].values
    max_val = np.max(y)

    # Detectar picos principales
    peaks, _ = find_peaks(
        y,
        height=factor_height * max_val,
        prominence=factor_prom * max_val,
        distance=2
    )

    pico_indices = []
    for pk in peaks:
        altura_max = y[pk]
        umbral_bajo = factor_bajo * altura_max
        # Izquierda
        left = pk
        while left > 0 and y[left] > umbral_bajo:
            pico_indices.append(left)
            left -= 1
        # Centro
        pico_indices.append(pk)
        # Derecha
        right = pk
        while right < len(y)-1 and y[right] > umbral_bajo:
            pico_indices.append(right)
            right += 1

    return sorted(set(pico_indices))

# Función para generar 2.a.pdf
def plot_remove_peaks(Mo, Rh, W):
    elementos = {"Molibdeno (Mo)": Mo, "Rodio (Rh)": Rh, "Tungsteno (W)": W}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # cada 2 kV

        colores = plt.cm.plasma(np.linspace(0, 1, len(voltajes_filtrados)))

        for color, kv in zip(colores, voltajes_filtrados):
            df = dicc[kv]
            pico_indices = detectar_zona_picos(df)

            # Graficar espectro original
            ax.plot(df["Energía"], df["Fotones"], color=color, linewidth=1.2)
            # Marcar puntos de pico
            ax.plot(df["Energía"].iloc[pico_indices],
                    df["Fotones"].iloc[pico_indices],
                    'ro', markersize=2)

        ax.set_title(nombre, fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    plt.subplots_adjust(top=1.0, bottom=0.1, wspace=0.15)
    plt.savefig("2.a.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

# Generar el punto 2.a
plot_remove_peaks(Mo, Rh, W)
