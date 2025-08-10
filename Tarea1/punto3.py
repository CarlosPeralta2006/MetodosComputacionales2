from Punto1 import data
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from punto2 import rem_picos
from punto2 import spline_diccionario
spectra = data()
Mo, Rh, W = spectra   # tomar la info original

# info para graficar
Mo_continuo, Mo_picos = rem_picos(Mo)
Rh_continuo, Rh_picos = rem_picos(Rh)
W_continuo, W_picos   = rem_picos(W)

dic_originales = {"Mo": Mo, "Rh": Rh, "W": W}
dic_continuos = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}
dic_picos = {"Mo": Mo_picos, "Rh": Rh_picos, "W": W_picos}

Mo_inter = spline_diccionario(Mo_continuo)
Rh_inter = spline_diccionario(Rh_continuo)
W_inter = spline_diccionario(W_continuo)

    
dic_continuo = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}    
dic_spline   = {"Mo": Mo_inter,   "Rh": Rh_inter,   "W": W_inter}






# --- Función para obtener solo los picos (restar spline al original) ---
def obtener_picos_restantes(dic_originales, dic_spline):
    dic_picos_restantes = {}
    for elem in dic_originales.keys():
        dic_orig = dic_originales[elem]
        dic_spl = dic_spline[elem]
        dic_picos_elem = {}
        for kv in dic_orig.keys():
            df_orig = dic_orig[kv]
            df_spl = dic_spl[kv]
            fotones_picos = df_orig["Fotones"].values - df_spl["Fotones"].values
            fotones_picos = np.clip(fotones_picos, 0, None)
            df_picos = df_orig.copy()
            df_picos["Fotones"] = fotones_picos
            dic_picos_elem[kv] = df_picos
        dic_picos_restantes[elem] = dic_picos_elem
    return dic_picos_restantes

dic_originales = {"Mo": Mo, "Rh": Rh, "W": W}
dic_spline = {"Mo": Mo_inter, "Rh": Rh_inter, "W": W_inter}
picos_restantes = obtener_picos_restantes(dic_originales, dic_spline)

# --- Función para graficar picos con zoom en x y alternancia de colores ---
def graficar_picos_zoom(picos_restantes, xlims, filename="3.a.pdf"):
    elementos = {"Molibdeno (Mo)": picos_restantes["Mo"],
                 "Rodio (Rh)": picos_restantes["Rh"],
                 "Tungsteno (W)": picos_restantes["W"]}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    colores = ["#FF00002E", "#00000092"]  # rojo fosforescente y negro

    for ax, (nombre, dicc) in zip(axes, elementos.items()):
        voltajes = sorted(dicc.keys())
        for i, kv in enumerate(voltajes):
            df = dicc[kv]
            ax.plot(df["Energía"], df["Fotones"], label=kv, color=colores[i % 2], linewidth=1.5)

        ax.set_title(nombre, fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.set_xlim(xlims[nombre])
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10, frameon=False)

    plt.subplots_adjust(top=1.0, bottom=0.25, wspace=0.15)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()
# --- Definir límites x para zoom en cada subplot (ajusta según tus picos) ---
xlims_zoom = {
    "Molibdeno (Mo)": (16.5, 20.5),
    "Rodio (Rh)": (19.5, 23.4),
    "Tungsteno (W)": (17, 21)
}

# --- Ejecutar la gráfica ---
graficar_picos_zoom(picos_restantes, xlims_zoom)



