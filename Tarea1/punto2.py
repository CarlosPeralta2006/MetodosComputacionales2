from Punto1 import data
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spectra = data()
Mo, Rh, W = spectra   # tomar la info original

# - - - PUNTO 2.a - - - 

def rem_picos(dic_elemento, prominence=0.13, ancho=2):  # valores óptimos por tanteo
    
    dict_continuo = {}  # dict que tendrá los puntos "continuos"
    dict_picos = {}     # dict que tendrá los puntos que se eliminarán pertenecientes a picos

    for kv, df in dic_elemento.items():  # estructura de los diccionarios: llave: hilovoltios, Info: df con energía y fotones
        x = df["Energía"].values
        y = df["Fotones"].values

        prom = prominence * np.max(y)
        peaks, _ = find_peaks(y, prominence=prom)

        df_fil = df.copy()  # copia
        picos_guardados = []

        for p in peaks:
            left_idx = max(p - ancho, 0)
            right_idx = min(p + ancho, len(y) - 1)  # análisis alrededor 

            # guardar valores eliminados
            for idx in range(left_idx, right_idx + 1):
                picos_guardados.append({"Energía": x[idx], "Fotones": y[idx]})

            # eliminar del continuo
            df_fil.loc[left_idx:right_idx, "Fotones"] = np.nan  # asigna valor NaN (usado luego)

        # guardar en los nuevos diccionarios
        dict_continuo[kv] = df_fil
        dict_picos[kv] = pd.DataFrame(picos_guardados)

    return dict_continuo, dict_picos

# info para graficar
Mo_continuo, Mo_picos = rem_picos(Mo)
Rh_continuo, Rh_picos = rem_picos(Rh)
W_continuo, W_picos   = rem_picos(W)

def graficar_2a(dic_originales, dic_continuos, dic_picos):
    elementos = ["Mo", "Rh", "W"]
    titulos = {"Mo": "Molibdeno (Mo)", "Rh": "Rodio (Rh)", "W": "Tungsteno (W)"}
    
    kv_labels = {"Mo": "34kV", "Rh": "41kV", "W": "16kV"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, elem in zip(axes, elementos):
        dic_orig = dic_originales[elem]
        dic_cont = dic_continuos[elem]
        dic_peak = dic_picos[elem]

        df_orig = dic_orig[kv_labels[elem]]
        df_cont = dic_cont[kv_labels[elem]]
        df_peak = dic_peak[kv_labels[elem]]

        ax.plot(df_orig["Energía"], df_orig["Fotones"], color="black", lw=1)
        ax.scatter(df_peak["Energía"], df_peak["Fotones"], color="red", s=15, label="Picos removidos")

        ax.set_title(f"{titulos[elem]} ({kv_labels[elem]})", fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11)

    plt.subplots_adjust(top=1, bottom=0.15, wspace=0.15)
    plt.savefig("2.a.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

dic_originales = {"Mo": Mo, "Rh": Rh, "W": W}
dic_continuos = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}
dic_picos = {"Mo": Mo_picos, "Rh": Rh_picos, "W": W_picos}

graficar_2a(dic_originales, dic_continuos, dic_picos)

# - - - PUNTO 2.b - - -

def spline_diccionario(dic_continuo, s_factor=0.02, n_fine=2000):
    dic_spline = {}
    for kv, df_cont in dic_continuo.items():
        mask = df_cont["Fotones"].notna()
        x_valid = df_cont["Energía"][mask].values
        y_valid = df_cont["Fotones"][mask].values

        if len(x_valid) < 4:
            dic_spline[kv] = pd.DataFrame({"Energía": df_cont["Energía"], "Fotones": [np.nan]*len(df_cont)})
            continue

        # Ajuste spline suavizado
        s_val = s_factor * len(x_valid)  # proporcional a número de puntos
        spline = UnivariateSpline(x_valid, y_valid, s=s_val)

        # Evaluar sobre la grilla original (para comparación directa)
        y_spline = spline(df_cont["Energía"].values)

        dic_spline[kv] = pd.DataFrame({
            "Energía": df_cont["Energía"].values,
            "Fotones": y_spline
        })
    return dic_spline

Mo_inter = spline_diccionario(Mo_continuo)
Rh_inter = spline_diccionario(Rh_continuo)
W_inter = spline_diccionario(W_continuo)

def graficar_2b(dic_continuo, dic_spline, filename="2.b.pdf", step=2):
    elementos = ["Mo", "Rh", "W"]
    titulos = {"Mo": "Molibdeno (Mo)", "Rh": "Rodio (Rh)", "W": "Tungsteno (W)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    kv_labels = {"Mo": "34kV", "Rh": "41kV", "W": "16kV"}
    
    for ax, elem in zip(axes, elementos):
        dic_cont = dic_continuos[elem]
        dic_spl  = dic_spline[elem]

        df_cont = dic_cont[kv_labels[elem]]
        df_spl  = dic_spl[kv_labels[elem]]
        
        #ax.scatter(df_cont["Energía"], df_cont["Fotones"], color="orange", s=12, label="Continuo sin picos")
        ax.plot(df_spl["Energía"], df_spl["Fotones"], color="blue", lw=1.5, label="Continuo sin picos")

        ax.set_title(f"{titulos[elem]} ({kv_labels[elem]})", fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10)

    plt.subplots_adjust(top=1, bottom=0.15, wspace=0.15)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    
dic_continuo = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}    
dic_spline   = {"Mo": Mo_inter,   "Rh": Rh_inter,   "W": W_inter}

graficar_2b(dic_continuo, dic_spline, step=2)