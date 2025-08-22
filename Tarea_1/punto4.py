import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import UnivariateSpline

from Punto1 import data
from punto2 import rem_picos, spline_diccionario
from punto3 import obtener_picos_restantes

# =========================
# 1. Cargar y procesar datos
# =========================
spectra = data()
Mo, Rh, W = spectra

dic_originales = {"Mo": Mo, "Rh": Rh, "W": W}

Mo_continuo, Mo_picos = rem_picos(Mo)
Rh_continuo, Rh_picos = rem_picos(Rh)
W_continuo, W_picos   = rem_picos(W)

dic_continuo = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}
Mo_inter = spline_diccionario(Mo_continuo)
Rh_inter = spline_diccionario(Rh_continuo)
W_inter = spline_diccionario(W_continuo)
dic_spline = {"Mo": Mo_inter, "Rh": Rh_inter, "W": W_inter}

picos_restantes = obtener_picos_restantes(dic_originales, dic_spline)

# ==============================================
# 2. Función de integración por cuadratura Gauss
# ==============================================
def integral_gauss_sobre_df(df, n=64, s_spline=0):
    x = df["Energía"].values
    y = df["Fotones"].values
    a, b = x.min(), x.max()
    spline = UnivariateSpline(x, y, s=s_spline)
    t, w = leggauss(n)
    E = 0.5*(b-a)*t + 0.5*(b+a)
    return 0.5*(b-a) * np.sum(w * spline(E))

# ===========================================
# 3. Verificación de normalización y comparación
# ===========================================
print("\n--- Comparación de integrales (deben ≈ 100) ---")
for elem in dic_originales:
    for kv, df in dic_originales[elem].items():
        trap = np.trapz(df["Fotones"], df["Energía"])
        simp = simpson(df["Fotones"], df["Energía"])
        gaus = integral_gauss_sobre_df(df, n=64)

        # Diferencias absolutas respecto a 100
        diffs = {
            "Trapezoide": abs(trap - 100),
            "Simpson": abs(simp - 100),
            "Gauss": abs(gaus - 100)
        }
        mejor = min(diffs, key=diffs.get)

        print(f"{elem} {kv}: Trap={trap:.6f}, Simpson={simp:.6f}, Gauss={gaus:.6f} → Mejor: {mejor} (dif={diffs[mejor]:.6f})")

# ================================================================
# 4. Calcular porcentajes área de picos respecto al continuo (Gauss)
# ================================================================
def calcular_porcentajes_gauss(dic_spline, picos_restantes):
    dic_porcentajes = {}
    for elem in dic_spline.keys():
        porcentajes_elem = []
        for kv in dic_spline[elem].keys():
            df_cont = dic_spline[elem][kv]
            df_peak = picos_restantes[elem][kv]
            area_cont = integral_gauss_sobre_df(df_cont)
            area_peak = integral_gauss_sobre_df(df_peak)
            pct_peak = 100 * area_peak / (area_cont + area_peak) if (area_cont + area_peak) > 0 else np.nan
            porcentajes_elem.append({
                "kV": kv,
                "kV_num": float(kv.replace("kV", "")),
                "pct_picos_gauss": pct_peak
            })
        df_elem = pd.DataFrame(porcentajes_elem).sort_values("kV_num").reset_index(drop=True)
        dic_porcentajes[elem] = df_elem
    return dic_porcentajes

dic_porcentajes_gauss = calcular_porcentajes_gauss(dic_spline, picos_restantes)

# ===========================
# 5. Graficar y guardar en 4.pdf
# ===========================
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}
marcas  = {"Mo": "o", "Rh": "s", "W": "^"}

plt.figure(figsize=(8,6))
for elem in dic_porcentajes_gauss.keys():
    df = dic_porcentajes_gauss[elem]
    plt.plot(df["kV_num"], df["pct_picos_gauss"], marker=marcas[elem],
             color=colores[elem], label=elem, lw=1.5)
plt.xlabel("Voltaje del tubo (kV)")
plt.ylabel("Porcentaje área picos / continuo (%)")
plt.title("Área picos respecto al continuo (Gauss–Legendre)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("4.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()