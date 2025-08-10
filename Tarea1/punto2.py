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
    
    # kV que quieres usar para cada elemento
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

def spline_diccionario(dic_continuo, s_factor=0.03):
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

# - - - PUNTO 2.c - - - 

def metricas(dic_spline):
    result = {}
    
    for elem, dic_spl in dic_spline.items():  # itera sobre los elementos
        datos = []
        for kv, df in dic_spl.items():  # obtiene los valores kV y DataFrame con Energía y Fotones
            x = df["Energía"].values
            y = df["Fotones"].values
            
            spl = UnivariateSpline(x,y,s=0)
            
            x_new = np.linspace(x.min(), x.max(), 5000)
            y_new = spl(x_new)  # evaluar el spline en la nueva grilla
            
            # máximo 
            y_max = np.max(y_new)
            indice_max = np.argmax(y_new)  # indice del valor máximo en y_new
            x_max = x_new[indice_max]
            
            # FWHM
            
            half_max = y_max / 2
            
            cruces = np.where(np.diff(np.sign(y_new - half_max)) != 0)[0]  
            
            if len(cruces) >= 2:
                x_izq = x_new[cruces[0]]
                x_der = x_new[cruces[-1]]
                fwhm = x_der - x_izq
            else:
                fwhm = np.nan 
            datos.append({
                "kV": kv, "y_max": y_max, "x_max":x_max, "FWHM": fwhm
            })
            df_result = pd.DataFrame(datos)
        try:
            df_result["kV_num"] = df_result["kV"].str.replace("kV", "").astype(float)
            df_result = df_result.sort_values("kV_num")
        except:
            pass
        result[elem] = df_result.reset_index(drop=True)

    return result

resultados_metricas = metricas(dic_spline)

def graficar_2c(resultados_metricas):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}

    ax = axes[0, 0]
    for elem, df in resultados_metricas.items():
        ax.plot(df["kV_num"], df["y_max"], marker="o", label=elem, color=colores[elem])
    ax.set_title("Altura máxima vs Voltaje del tubo")
    ax.set_xlabel("Voltaje (kV)")
    ax.set_ylabel("Conteo de fotones máximo (u.a.)")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax = axes[0, 1]
    for elem, df in resultados_metricas.items():
        ax.plot(df["kV_num"], df["x_max"], marker="o", label=elem, color=colores[elem])
    ax.set_title("Energía del máximo vs Voltaje del tubo")
    ax.set_xlabel("Voltaje (kV)")
    ax.set_ylabel("Energía del máximo (keV)")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax = axes[1, 0]
    for elem, df in resultados_metricas.items():
        ax.plot(df["kV_num"], df["FWHM"], marker="o", label=elem, color=colores[elem])
    ax.set_title("FWHM vs Voltaje del tubo")
    ax.set_xlabel("Voltaje (kV)")
    ax.set_ylabel("FWHM (keV)")
    ax.grid(True, linestyle="--", alpha=0.6)

    ax = axes[1, 1]
    for elem, df in resultados_metricas.items():
        ax.plot(df["x_max"], df["y_max"], marker="o", label=elem, color=colores[elem])
    ax.set_title("Altura máxima vs Energía del máximo")
    ax.set_xlabel("Energía del máximo (keV)")
    ax.set_ylabel("Conteo de fotones máximo (u.a.)")
    ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12)
    plt.subplots_adjust(top=1, bottom=0.15, wspace=0.25)
    plt.savefig("2.c.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    
graficar_2c(resultados_metricas)
            