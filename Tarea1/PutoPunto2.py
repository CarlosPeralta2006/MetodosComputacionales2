from Punto1 import data
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spectra = data()
Mo, Rh, W = spectra   # tomar la info original

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
            df_fil.loc[left_idx:right_idx, "Fotones"] = np.nan

        # guardar en los nuevos diccionarios
        dict_continuo[kv] = df_fil
        dict_picos[kv] = pd.DataFrame(picos_guardados)

    return dict_continuo, dict_picos

# info para graficar
Mo_continuo, Mo_picos = rem_picos(Mo)
Rh_continuo, Rh_picos = rem_picos(Rh)
W_continuo, W_picos   = rem_picos(W)

def graficar_2a(dic_originales, dic_continuos, dic_picos):
    # estructura
    elementos = ["Mo", "Rh", "W"]
    titulos = {"Mo": "Molibdeno (Mo)", "Rh": "Rodio (Rh)", "W": "Tungsteno (W)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, elem in zip(axes, elementos):
        dic_orig = dic_originales[elem]
        dic_cont = dic_continuos[elem]
        dic_peak = dic_picos[elem]

        # elección de un voltaje representativo (el del medio) o ponerlo manual
        kvs = sorted(dic_orig.keys())
        kv = kvs[len(kvs)//2]  
        # datos de ese kV seleccionado
        df_orig = dic_orig[kv]
        df_cont = dic_cont[kv]
        df_peak = dic_peak[kv]

        ax.plot(df_orig["Energía"], df_orig["Fotones"], color="black", lw=1)
        ax.scatter(df_peak["Energía"], df_peak["Fotones"], color="red", s=15, label="Picos removidos")

        ax.set_title(f"{titulos[elem]} - {kv}", fontsize=14, weight="bold")
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