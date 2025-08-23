import pandas as pd
import numpy as np
from scipy.signal import savgol_filter 
import matplotlib.pyplot as plt

# - - - 2.a - - - - - - - - - 

data = pd.read_csv("Data2/tomography_data/SN_d_tot_V2.0.csv")
data["spots"] = data["spots"].replace(-1, pd.NA)
data["spot_std"] = data["spot_std"].replace(-1.0, pd.NA)

data.loc[0, "spots"] = data["spots"].mean()
data.loc[0, "spot_std"] = data["spot_std"].mean()

data["spots"] = pd.to_numeric(data["spots"])
data["spot_std"] = pd.to_numeric(data["spot_std"])

# new data
data["spots"] = data["spots"].interpolate(method="pchip")
data["spot_std"] = data["spot_std"].interpolate(method="pchip")

spots = data["spots"]

# si solo se quitan, la periodicidad del muestreo se perdería

# - - - 2.b - - - - - - - - - 

# implementación de la Transformada Rápida de Fourier
param = 10
F = np.fft.fft(spots, n = param * len(spots))
freq = np.fft.fftfreq(n = param * len(spots), d = 1)


mask = freq > 1/100000  # evitar que tome el pico ubicado en 0 (promedio) y val. negativos
freq_pos = freq[mask]
F_pos = F[mask]

idx_max = np.argmax(np.abs(F_pos))  # indica en qué índice se encuentra el valor máximo
f_dom = freq_pos[idx_max]
T = 1 / f_dom

with open("2.b.txt", "w") as f:
    f.write(f"Periodo: {T:.2f}\n")
    
print(T)


# filtro pasa bajas desde el espacio de tiempos
ts = np.arange(len(spots))
spots_savgol = savgol_filter(spots, window_length=500, polyorder=5)

plt.figure(figsize=(12,6))
plt.legend()
plt.plot(ts, spots, color='red', label='Datos originales')
plt.plot(ts, spots_savgol, color='b', label='Datos filtrados')   
plt.xlabel('Tiempo (días)')
plt.ylabel('Número de Manchas Solares')
plt.title('Eliminación del Ruido de los Datos')

plt.savefig('2.b.data.pdf', bbox_inches="tight", pad_inches=0.1)




