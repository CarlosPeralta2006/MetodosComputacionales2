import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Punto1 import data

# Obtener los datos
spectra = data()
Mo, Rh, W = spectra

# Obtener el primer DataFrame de Mo (10kV)
primer_kv = '30kV'  # Especificamos directamente la clave del primer DataFrame
df_primer = Mo[primer_kv]

# Graficar este DataFrame específico
plt.figure(figsize=(10, 6))
plt.plot(df_primer['Energía'], df_primer['Fotones'], 'b-', linewidth=1.5)
plt.title(f'Espectro de Molibdeno (Mo) a {primer_kv}', fontsize=14)
plt.xlabel('Energía (keV)', fontsize=12)
plt.ylabel('Conteo de fotones (u.a.)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xlim(0, 35) 
# Añadir texto con información sobre los datos
plt.text(0.5, 0.95, f'Datos: {len(df_primer)} puntos', 
         transform=plt.gca().transAxes, ha='center', 
         bbox=dict(facecolor='white', alpha=0.8))

# Mostrar y guardar el gráfico
plt.tight_layout()
plt.savefig(f'molibdeno_{primer_kv}.pdf', bbox_inches='tight')
plt.show()

# Mostrar información del DataFrame
print(f"\nInformación del DataFrame ({primer_kv}):")
print(f"Número de filas: {len(df_primer)}")
print(f"Rango de energía: {df_primer['Energía'].min()} - {df_primer['Energía'].max()} keV")
print("\nPrimeras 5 filas:")
print(df_primer.head())
print("\nÚltimas 5 filas:")
print(df_primer.tail())