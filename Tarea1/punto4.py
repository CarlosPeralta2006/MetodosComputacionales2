import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Punto1 import data

spectra = data()
Mo, Rh, W = spectra

# Función mejorada para identificación de picos
def identify_main_peaks(df, threshold_factor=0.3):
    peaks = []
    edges = []
    
    # Calcular umbral como fracción del máximo
    threshold = df['Fotones'].max() * threshold_factor
    
    # Identificar puntos sobre el umbral
    above_threshold = df['Fotones'] > threshold
    
    # Encontrar regiones continuas sobre el umbral
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Si no hay picos detectados
    if len(starts) == 0:
        return [], []
        
    # Asegurar que starts y ends tengan la misma longitud
    if starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)
    if len(starts) > len(ends):
        ends = np.append(ends, len(df)-1)
    
    # Para cada región, encontrar el pico máximo y bordes
    for start, end in zip(starts, ends):
        peak_index = df.iloc[start:end]['Fotones'].idxmax()
        peaks.append(peak_index)
        edges.append((start, end))
    
    return peaks, edges

# Configurar gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)

# Parámetros de voltaje
kvs = ['27kV', '10kV', '38kV']
colors = ['b', 'r', 'g']
data_dict = {'27kV': Mo['27kV'], '10kV': Rh['10kV'], '38kV': W['38kV']}

for ax, kv, color in zip(axes, kvs, colors):
    df = data_dict[kv]
    
    # Graficar la curva principal
    ax.plot(df['Energía'], df['Fotones'], f'{color}-', linewidth=1.5)
    ax.set_title(f'Espectro a {kv}', fontsize=14)
    ax.set_xlabel('Energía (keV)', fontsize=12)
    if kv == '27kV':  # Solo mostrar etiqueta Y en el primer gráfico
        ax.set_ylabel('Conteo de fotones (u.a.)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 35)
    
    # Solo marcar picos si no es 10kV
    if kv != '10kV':
        peaks, edges = identify_main_peaks(df, threshold_factor=0.25)
        for peak, (start, end) in zip(peaks, edges):
            ax.plot(df['Energía'].iloc[peak], df['Fotones'].iloc[peak], 
                   'ro', markersize=8, markeredgecolor='k')
            ax.plot(df['Energía'].iloc[start], df['Fotones'].iloc[start], 
                   'yo', markersize=8, markeredgecolor='k')
            ax.plot(df['Energía'].iloc[end], df['Fotones'].iloc[end], 
                   'yo', markersize=8, markeredgecolor='k')

plt.tight_layout()
plt.savefig("espectros_con_picos.pdf", bbox_inches='tight', dpi=200)
plt.show()
