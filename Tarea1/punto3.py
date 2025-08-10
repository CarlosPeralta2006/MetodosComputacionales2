import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Punto1 import data

# Obtener los datos
spectra = data()
Mo, Rh, W = spectra

# Configurar 3 subplots (1 fila x 3 columnas)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 18 pulgadas de ancho x 6 de alto
plt.subplots_adjust(wspace=0.3)  # Espacio entre gráficos

# kV a graficar (el mismo para los 3 elementos)
kV = '30kV'

# ---- Gráfico 1: Molibdeno (Mo) ----
axes[0].plot(Mo[kV]['Energía'], Mo[kV]['Fotones'], 'b-', linewidth=1.5)
axes[0].set_title(f'Molibdeno (Mo) a {kV}', fontsize=14)
axes[0].set_xlabel('Energía (keV)', fontsize=12)
axes[0].set_ylabel('Conteo de fotones (u.a.)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_xlim(0, 35)
axes[0].text(0.5, 0.95, f'Datos: {len(Mo[kV])} puntos', 
            transform=axes[0].transAxes, ha='center', 
            bbox=dict(facecolor='white', alpha=0.8))

# ---- Gráfico 2: Rodio (Rh) ----
axes[1].plot(Rh[kV]['Energía'], Rh[kV]['Fotones'], 'r-', linewidth=1.5)
axes[1].set_title(f'Rodio (Rh) a {kV}', fontsize=14)
axes[1].set_xlabel('Energía (keV)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_xlim(0, 35)
axes[1].text(0.5, 0.95, f'Datos: {len(Rh[kV])} puntos', 
            transform=axes[1].transAxes, ha='center', 
            bbox=dict(facecolor='white', alpha=0.8))

# ---- Gráfico 3: Tungsteno (W) ----
axes[2].plot(W[kV]['Energía'], W[kV]['Fotones'], 'g-', linewidth=1.5)
axes[2].set_title(f'Tungsteno (W) a {kV}', fontsize=14)
axes[2].set_xlabel('Energía (keV)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].set_xlim(0, 35)
axes[2].text(0.5, 0.95, f'Datos: {len(W[kV])} puntos', 
            transform=axes[2].transAxes, ha='center', 
            bbox=dict(facecolor='white', alpha=0.8))

# Guardar y mostrar
plt.tight_layout()
plt.savefig(f'espectros_comparativos_{kV}.pdf', bbox_inches='tight')
plt.show()

# Mostrar información de los DataFrames
for elemento, df in zip(['Mo', 'Rh', 'W'], [Mo[kV], Rh[kV], W[kV]]):
    print(f"\nInformación del DataFrame ({elemento} {kV}):")
    print(f"Número de filas: {len(df)}")
    print(f"Rango de energía: {df['Energía'].min()} - {df['Energía'].max()} keV")