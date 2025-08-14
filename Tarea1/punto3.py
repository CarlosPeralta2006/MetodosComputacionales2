from Punto1 import data                           #función que lee todos los espectros de Mo, Rh y W desde los archivos .dat y devuelve tres diccionarios.
from scipy.signal import find_peaks               #find_peaks: de scipy.signal, para detectar posiciones de picos.
from scipy.interpolate import UnivariateSpline    #UnivariateSpline: para interpolación suave de datos.
import matplotlib.pyplot as plt
import numpy as np      
import pandas as pd                                #para manipular tablas de datos.
from scipy.optimize import curve_fit               #curve_fit: de scipy.optimize, para ajustar funciones a datos (en este caso, gaussianas).
from punto2 import rem_picos                       #rem_picos y spline_diccionario: funciones definidas en punto2 para limpiar picos y reconstruir el continuo.
from punto2 import spline_diccionario

spectra = data()                        #Llama a data() para obtener tres diccionarios:          
Mo, Rh, W = spectra                     # Mo, Rh, W: cada uno tiene claves tipo "34kV" y valores DataFrames con columnas "Energía" y "Fotones".


# info para graficar
Mo_continuo, Mo_picos = rem_picos(Mo)       #continuo: espectro sin los picos (valores reemplazados por NaN).
Rh_continuo, Rh_picos = rem_picos(Rh)       #picos: solo los puntos detectados como picos.
W_continuo, W_picos   = rem_picos(W)

dic_originales = {"Mo": Mo, "Rh": Rh, "W": W}       #Agrupa en diccionarios para manipularlos más fácilmente.
dic_continuos = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}
dic_picos = {"Mo": Mo_picos, "Rh": Rh_picos, "W": W_picos}

Mo_inter = spline_diccionario(Mo_continuo)   #spline_diccionario rellena los huecos del continuo usando splines, produciendo una curva suave que representa el fondo sin picos.
Rh_inter = spline_diccionario(Rh_continuo)
W_inter = spline_diccionario(W_continuo)
    
dic_continuo = {"Mo": Mo_continuo, "Rh": Rh_continuo, "W": W_continuo}    #Se crean dos diccionarios: uno con el continuo limpio (dic_continuo) y otro con el spline ajustado (dic_spline).
dic_spline   = {"Mo": Mo_inter,   "Rh": Rh_inter,   "W": W_inter}


def obtener_picos_restantes(dic_originales, dic_spline):    #Resta el continuo interpolado al espectro original → se quedan solo las contribuciones de los picos.
    dic_picos_restantes = {}  #Crea un diccionario vacío donde se guardarán los espectros de picos para cada elemento.                      
    for elem in dic_originales.keys():   #Recorre cada elemento químico presente ("Mo", "Rh", "W").
        dic_orig = dic_originales[elem] #dic_orig: espectros originales de ese elemento (clave = voltaje, valor = DataFrame con Energía/Fotones).
        dic_spl = dic_spline[elem]          #dic_spl: espectros del continuo spline para ese mismo elemento.
        dic_picos_elem = {}        # Crea un diccionario para guardar, solo para este elemento, sus espectros de picos separados por voltaje.
        for kv in dic_orig.keys(): #Recorre todos los voltajes (kv) disponibles para ese elemento, por ejemplo "34kV", "41kV".
            df_orig = dic_orig[kv]  #df_orig: DataFrame del espectro original (incluye picos y continuo) a ese voltaje.
            df_spl = dic_spl[kv]
            fotones_picos = df_orig["Fotones"].values - df_spl["Fotones"].values  #Calcula la diferencia punto a punto entre el original y el continuo spline.
            fotones_picos = np.clip(fotones_picos, 0, None)   #np.clip(..., 0, None): asegura que valores negativos (ruido) se pongan a 0.; #Esto elimina ruido que podría dar diferencias negativas por errores de ajuste.
            df_picos = df_orig.copy()  #Copia el DataFrame original (para mantener columnas como "Energía" y otros metadatos).
            df_picos["Fotones"] = fotones_picos  #Sobrescribe la columna "Fotones" con la versión que contiene solo la contribución de los picos (el continuo se eliminó).
            dic_picos_elem[kv] = df_picos  
        dic_picos_restantes[elem] = dic_picos_elem  # Resta el continuo interpolado al espectro original → se quedan solo las contribuciones de los picos.
    return dic_picos_restantes   #Devuelve un diccionario anidado: elemento → voltaje → DataFrame con solo picos.                        
picos_restantes = obtener_picos_restantes(dic_originales, dic_spline)


def graficar_picos_zoom(picos_restantes, xlims, filename="3.a.pdf"):        #picos_restantes: diccionario anidado con los espectros de solo picos para Mo, Rh y W.
   
    elementos = {"Molibdeno (Mo)": picos_restantes["Mo"],                   #Colorea alternadamente según voltaje.
                 "Rodio (Rh)": picos_restantes["Rh"],                       #xlims es un diccionario con los rangos de energía donde se quiere hacer zoom.
                 "Tungsteno (W)": picos_restantes["W"]}                     # Dibuja los picos para cada elemento en subplots separados.

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)    #Prepara una figura con 1 fila y 3 columnas de subplots (uno para cada elemento).
                                                                    #sharey=True: todos los subplots comparten el mismo eje Y.
    colores = ["#FF00002E", "#00000092"]  #Lista de colores que se usarán de forma alternada para diferenciar las curvas de distintos voltajes.

    for ax, (nombre, dicc) in zip(axes, elementos.items()):  #Cada eje de la figura (ax).El nombre del elemento (nombre) y su diccionario de espectros (dicc).
        voltajes = sorted(dicc.keys()) #Obtiene y ordena alfabéticamente las claves (voltajes) del elemento, por ejemplo "30kV", "32kV", etc.
        for i, kv in enumerate(voltajes):
            df = dicc[kv]   #Obtiene el DataFrame correspondiente a ese voltaje (kv), que contiene "Energía" y "Fotones" solo de los picos.
            ax.plot(df["Energía"], df["Fotones"], label=kv, color=colores[i % 2], linewidth=0.8) #Color alternado (i % 2 elige entre rojo y negro).

        ax.set_title(nombre, fontsize=14, weight="bold")
        ax.set_xlabel("Energía (keV)", fontsize=12)
        ax.set_xlim(xlims[nombre])
        ax.grid(True, linestyle="--", alpha=0.6)  #Activa la cuadrícula con línea discontinua y transparencia del 60%.

    axes[0].set_ylabel("Conteo de fotones (u.a.)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels() #Recupera los elementos y etiquetas para construir una leyenda
    

    plt.subplots_adjust(top=1.0, bottom=0.25, wspace=0.15)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

xlims_zoom = {
    "Molibdeno (Mo)": (16.5, 20.5),
    "Rodio (Rh)": (19.5, 23.4),
    "Tungsteno (W)": (7.5, 13)}

graficar_picos_zoom(picos_restantes, xlims_zoom)


def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) #sigma = desviación estándar (anchura del pico).

resultados = {elem: [] for elem in picos_restantes.keys()}  #Crea un diccionario vacío donde cada clave es un elemento ("Mo", "Rh", "W") y el valor asociado es una lista donde se guardarán los resultados del ajuste gaussiano para cada voltaje.

for elem, dic_voltajes in picos_restantes.items():
    for voltaje, df in dic_voltajes.items():     #Itera sobre cada voltaje (voltaje, por ejemplo "34kV") y su respectivo DataFrame df con columnas "Energía" y "Fotones".            
        x = df["Energía"].values
        y = df["Fotones"].values  #Extrae los valores numéricos de energía y fotones del DataFrame como arreglos numpy.

        # Encontrar pico más alto
        peaks, _ = find_peaks(y) #Usa find_peaks para encontrar todos los índices donde hay máximos locales en y.
        if len(peaks) == 0:
            continue  #Si no se encontró ningún pico, pasa al siguiente voltaje.
            
        idx_max = peaks[np.argmax(y[peaks])] #De todos los picos detectados, identifica el índice (idx_max) del pico con mayor altura.

        # Tomar ventana alrededor del pico Define un rango de datos alrededor del pico más alto para el ajuste:
        ancho_ventana = 5  # puntos antes y después
        i_min = max(0, idx_max - ancho_ventana) #max(0, ...) y min(len(x), ...) evitan salirse de los límites del array.
        i_max = min(len(x), idx_max + ancho_ventana + 1)

        x_fit = x[i_min:i_max]
        y_fit = y[i_min:i_max]  #Extrae los datos de energía y fotones dentro de la ventana seleccionada para el ajuste.

        # Estimaciones iniciales para el ajuste
        A0 = y[idx_max]   #Calcula valores iniciales para el ajuste gaussiano
        x0_0 = x[idx_max]
        sigma0 = (x_fit.max() - x_fit.min()) / 4   #sigma0: anchura inicial aproximada, estimada como un cuarto del rango de x_fit.

        try:   
            #popt contendrá los valores óptimos encontrados para A, x0 y sigma.
            popt, _ = curve_fit(gauss, x_fit, y_fit, p0=[A0, x0_0, sigma0]) #Llama a curve_fit para ajustar la función gauss a los datos (x_fit, y_fit).
            A, x0, sigma = popt #p0 pasa las estimaciones iniciales.
            FWHM = 2.355 * abs(sigma)  # relación gaussiana  
            resultados[elem].append({"Voltaje": voltaje, "Altura": A, "Posicion": x0, "FWHM": FWHM}) #Guarda un diccionario con los resultados del ajuste para ese voltaje dentro de la lista del elemento:
        except RuntimeError:
            continue


df_resultados = {elem: pd.DataFrame(val) for elem, val in resultados.items()}  #Convierte el diccionario resultados (que tenía listas de diccionarios) en un nuevo diccionario df_resultados donde:

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


colores = {"Mo": "red", "Rh": "blue", "W": "green"}

for elem, df_res in df_resultados.items(): #Itera sobre cada elemento (elem) y su respectivo DataFrame (df_res).
    if df_res.empty:  
        
        #Si el DataFrame está vacío (no hubo picos ajustados para ese elemento), pasa al siguiente.
        continue
    df_res = df_res.sort_values("Voltaje")
    axes[0].plot(df_res["Voltaje"], df_res["Altura"], marker="o", color=colores[elem], label=elem)
    axes[1].plot(df_res["Voltaje"], df_res["FWHM"], marker="o", color=colores[elem], label=elem)

axes[0].set_xlabel("Voltaje del tubo (kV)")
axes[0].set_ylabel("Altura del pico (u.a.)")
axes[0].set_title("Altura vs Voltaje")
axes[0].grid(True)
axes[0].legend()

axes[1].set_xlabel("Voltaje del tubo (kV)")
axes[1].set_ylabel("Ancho a media altura (keV)")
axes[1].set_title("FWHM vs Voltaje")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("3.b.pdf", bbox_inches="tight")
plt.show()
