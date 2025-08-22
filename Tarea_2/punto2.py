import pandas as pd
from pandas import DataFrame as DF
import matplotlib.pyplot as plt

# - - - 2.a - - - - - - - - - 

data = pd.read_csv("Data2/tomography_data/SN_d_tot_V2.0.csv")
data["spots"] = data["spots"].replace(-1, pd.NA)
data["spot_std"] = data["spot_std"].replace(-1.0, pd.NA)

data["spots"][0] = data["spots"].mean()
data["spot_std"][0] = data["spot_std"].mean()

data["spots"] = pd.to_numeric(data["spots"])
data["spot_std"] = pd.to_numeric(data["spot_std"])

# new data
data["spots"] = data["spots"].interpolate(method="pchip")
data["spot_std"] = data["spot_std"].interpolate(method="pchip")

# si solo se quitan, la periodicidad del muestreo se perdería

# - - - 2.b - - - - - - - - - 