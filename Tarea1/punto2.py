from Punto1 import data
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter



data_copy = data()
Mo, Rh, W = data_copy

# Función para detectar EXTREMOS de picos (máximos locales)
def detectar_extremos_picos(energia, fotones, altura_min=0.1, distancia=10):
    """
    energia: array de energía (keV)
    fotones: array de conteo de fotones
    altura_min: altura mínima para considerar un pico (ajustar según datos)
    distancia: separación mínima entre picos (en puntos)
    """
    # Suavizar datos para mejor detección
    fotones_suavizados = savgol_filter(fotones, window_length=15, polyorder=2)
    
    # Encontrar picos (máximos locales)
    peaks, _ = find_peaks(fotones_suavizados, height=altura_min, distance=distancia)
    
    return energia[peaks], fotones[peaks]  # Posiciones y valores de los extremos

# Función principal para graficar
def plot_spectra_con_picos_extremos(Mo, Rh, W):
    elementos = {
        "Molibdeno (Mo)": Mo,
        "Rodio (Rh)": Rh,
        "Tungsteno (W)": W
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        voltajes_filtrados = voltajes[::2]  # Tomar kV de dos en dos
        
        # Graficar cada kV y sus picos
        for kv in voltajes_filtrados:
            df = dicc[kv]
            energia = df["Energía"].values
            fotones = df["Fotones"].values
            
            # Graficar línea principal (azul)
            ax.plot(energia, fotones, '-', linewidth=1.2, label=f'{kv}')
            
            # Detectar y graficar EXTREMOS de picos (rojos)
            peaks_x, peaks_y = detectar_extremos_picos(energia, fotones)
            ax.scatter(peaks_x, peaks_y, color='red', s=30, 
                       label='Picos' if kv == voltajes_filtrados[0] else "")
        
        ax.set_title(nombre, fontsize=14)
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    fig.legend(loc='lower center', ncol=len(voltajes_filtrados), fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("espectros_con_picos_extremos.pdf", bbox_inches='tight')
    plt.show()

# Cargar datos (usa tu función original)
Mo, Rh, W = data()

# Generar gráfico condensado
plot_spectra_con_picos_extremos(Mo, Rh, W)