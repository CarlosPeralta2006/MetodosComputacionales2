from Punto1 import data
import numpy as np

data_copy = data()
Mo, Rh, W = data_copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Función para DETECTAR picos (sin eliminarlos)
def detectar_picos(df):
    fotones = df['Fotones'].values
    suavizado = savgol_filter(fotones, window_length=15, polyorder=2)
    residuos = fotones - suavizado
    picos_mask = np.abs(residuos) > 3.0 * np.std(residuos)
    return picos_mask

# Función para GRAFICAR (igual que tu original, pero marcando picos)
def plot_spectra_con_picos(Mo, Rh, W):
    elementos = {"Molibdeno (Mo)": Mo, "Rodio (Rh)": Rh, "Tungsteno (W)": W}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # Tomar kV de dos en dos
        
        for kv in voltajes_filtrados:
            df = dicc[kv]
            picos_mask = detectar_picos(df)
            
            # Graficar línea principal (azul)
            ax.plot(df["Energía"], df["Fotones"], 'b-', linewidth=1.2)
            
            # Marcar picos en rojo (como en tu imagen)
            ax.scatter(df["Energía"][picos_mask], df["Fotones"][picos_mask], 
                       color='red', s=30, zorder=5, label='Picos' if kv == voltajes_filtrados[0] else "")
        
        ax.set_title(nombre, fontsize=14)
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    plt.legend()
    plt.savefig("picos_marcados.pdf", bbox_inches='tight')
    plt.show()

# Cargar datos (usa tu función original)
Mo, Rh, W = data()

# Generar gráficos
plot_spectra_con_picos(Mo, Rh, W)