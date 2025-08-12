import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from Punto1 import data
from punto2 import rem_picos
from punto2 import spline_diccionario
from punto3 import obtener_picos_restantes
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

picos_restantes = obtener_picos_restantes(dic_originales, dic_spline)

def calcular_porcentajes(dic_spline, picos_restantes):
    dic_porcentajes = {}
    for elem in dic_spline.keys():
        porcentajes_elem = []
        for kv in dic_spline[elem].keys():
            df_cont = dic_spline[elem][kv]
            df_peak = picos_restantes[elem][kv]
            
            x_cont = df_cont["Energía"].values
            y_cont = df_cont["Fotones"].values
            
            x_peak = df_peak["Energía"].values
            y_peak = df_peak["Fotones"].values
            
            # Integrales con trapecio
            area_cont_trap = np.trapezoid(y_cont, x_cont)
            area_peak_trap = np.trapezoid(y_peak, x_peak)
            
            #Integrales con simpson
            area_cont_simp = simpson(y_cont, x_cont) if len(x_cont) >= 3 else np.nan
            area_peak_simp = simpson(y_peak, x_peak) if len(x_peak) >= 3 else np.nan
            
            total_trap=area_cont_trap+area_peak_trap
            total_simp=area_cont_simp+area_peak_simp
            pct_peak_trap=100*area_peak_trap/total_trap
            pct_peak_simp=100*area_peak_simp/total_simp
            
            porcentajes_elem.append({
                "kV": kv,
                "kV_num": float(kv.replace("kV", "")),
                "pct_picos_trap": pct_peak_trap,
                "pct_picos_simp": pct_peak_simp
            })
        
        # Convertir a DataFrame y ordenar por kV_num
        df_elem = pd.DataFrame(porcentajes_elem).sort_values("kV_num").reset_index(drop=True)
        dic_porcentajes[elem] = df_elem
    
    return dic_porcentajes

dic_porcentajes=calcular_porcentajes(dic_spline,picos_restantes)

#Paleta
colores = {"Mo": "tab:blue", "Rh": "tab:orange", "W": "tab:green"}
marcas  = {"Mo": "o", "Rh": "s", "W": "^"}

#Grafico usando trapezoide
def graficar_trapecio(dic_porcentajes):
    plt.figure(figsize=(8,6))
    for elem in dic_porcentajes.keys():
        df = dic_porcentajes[elem]
        plt.plot(df["kV_num"], df["pct_picos_trap"], marker=marcas[elem],
                 color=colores[elem], label=elem, lw=1.5)
    plt.xlabel("Voltaje del tubo (kV)")
    plt.ylabel("Porcentaje área picos / continuo (%)")
    plt.title("Área picos respecto al continuo (Trapecio)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("4_trapecio.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

#Grafico usando Simpson
def graficar_simpson(dic_porcentajes):
    plt.figure(figsize=(8,6))
    for elem in dic_porcentajes.keys():
        df = dic_porcentajes[elem]
        plt.plot(df["kV_num"], df["pct_picos_simp"], marker=marcas[elem],
                 color=colores[elem], label=elem, lw=1.5)
    plt.xlabel("Voltaje del tubo (kV)")
    plt.ylabel("Porcentaje área picos / continuo (%)")
    plt.title("Área picos respecto al continuo (Simpson)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("4_simpson.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

#Grafico comparacion entre trapezoide y Simpson
def graficar_comparacion(dic_porcentajes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, elem in zip(axes, ["Mo", "Rh", "W"]):
        df = dic_porcentajes[elem]
        ax.plot(df["kV_num"], df["pct_picos_trap"], marker=marcas[elem],
                color=colores[elem], label="Trapecio", lw=1.5)
        ax.plot(df["kV_num"], df["pct_picos_simp"], marker=marcas[elem],
                color="black", linestyle="--", label="Simpson", lw=1.2)
        ax.set_title(elem, fontsize=14, weight="bold")
        ax.set_xlabel("Voltaje del tubo (kV)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Porcentaje área picos / continuo (%)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11)
    plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.15)
    plt.savefig("4_comparacion.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

#Plotear
graficar_trapecio(dic_porcentajes)
graficar_simpson(dic_porcentajes)
graficar_comparacion(dic_porcentajes)

for elem, df in dic_porcentajes.items():
    # Diferencia sin valor absoluto
    delta = df["pct_picos_trap"] - df["pct_picos_simp"]
    
    # Índice donde la diferencia absoluta es máxima
    idx_max = delta.abs().idxmax()
    
    diff_max = delta.iloc[idx_max]
    kv_max = df["kV"].iloc[idx_max]
    
    metodo = "Trapecio" if diff_max > 0 else "Simpson"
    
    print(elem + ": Diferencia máxima = " + str(round(abs(diff_max), 6)) + " % en" + str(kv_max) + " → " + metodo + " fue mayor")