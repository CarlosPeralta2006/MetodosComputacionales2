import pandas as pd
from pandas import DataFrame as DF
import numpy as np
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
ts = data["decimal_date"]


# si solo se quitan, la periodicidad del muestreo se perdería

# - - - 2.b - - - - - - - - - 

param = 6
F = np.fft.fft(spots, n = param * len(spots))
#F = np.fft.fftshift(F)

freq = np.fft.fftfreq(n = param * len(spots), d = 1)
#freq = np.fft.fftshift(freq)


F_filtrada = F.copy()
F_filtrada[abs(freq) > 0.05] = 0.

plt.plot(freq,abs(F))
plt.plot(freq,abs(F_filtrada))

plt.show()