import numpy as np
import matplotlib.pyplot as plt




def generate_data(tmax,dt,A,freq,noise):
     
    #Generates a sin wave of the given amplitude (A) and frequency (freq),
    #sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    #A random number with the given standard deviation (noise) is added to each data point.
    #Returns an array with the times and the measurements of the signal. 
        
     ts = np.arange(0,tmax+dt,dt)
     return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)
 
 
 
 

def Fourier_transform(t, y, freqs):

    N = len(t)
    F = np.zeros(len(freqs), dtype=complex)  # arreglo complejo para la transformada

   #Reserva un vector complejo F de longitud igual al número de frecuencias.Cada uno almancenando la transformada en la frecuencia freqs[k].
   
    for k, fk in enumerate(freqs):   
        expo = np.exp(-2j * np.pi * fk * t)  # e^{-2πi f t} es un vector (porque t es un array).
        F[k] = np.sum(y * expo)

    return F





#Datos a considerar(Arbitrarios)
tmax= 3
dt = 0.01 
A= 1
freq = 5
noise = 0.2

# Generar datos
t, y = generate_data(tmax, dt, A, freq, noise)
# Calcular frecuencia de Nyquist
f_nyq = 1/(2*dt)
# Construir frecuencias hasta 2.7 × Nyquist
freqs = np.linspace(0, 2.7*f_nyq, 2000)
# Calcular transformada de Fourier
F = Fourier_transform(t, y, freqs)
# Calcular la amplitud del espectro (módulo de la transformada)
amplitud_espectro = np.abs(F)
# Graficar la amplitud del espectro en función de la frecuencia
plt.figure(figsize=(12, 6))
# Gráfico principal: Amplitud del espectro
plt.plot(freqs, amplitud_espectro, 'b-', linewidth=1.5, label='Amplitud del espectro')
# Líneas de referencia importantes
plt.axvline(x=freq, color='red', linestyle='--', alpha=0.8, 
            label=f'Frecuencia de la señal ({freq} Hz)')
plt.axvline(x=f_nyq, color='green', linestyle='--', alpha=0.8, 
            label=f'Frecuencia de Nyquist ({f_nyq:.0f} Hz)')
plt.axvline(x=2.7*f_nyq, color='yellow', linestyle='--', alpha=0.8, linewidth=2.5,
            label=f'2.7 × Nyquist ({2.7*f_nyq:.0f} Hz)')
# Configuración del gráfico
plt.xlabel("Frecuencia (Hz)", fontsize=12)
plt.ylabel("Amplitud del espectro |F(f)|", fontsize=12)
plt.title("Amplitud del espectro de Fourier hasta 2.7 veces la frecuencia de Nyquist", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xlim(0, 2.7*f_nyq)

plt.tight_layout()
plt.savefig("1.a.pdf", bbox_inches='tight', dpi=300)
plt.show()




def calculate_SN_freq(amplitud_espectro, freq, freqs, window_size=5):
    """
    Calcula SN_freq: altura del pico principal dividida por la desviación estándar
    del fondo (parte sin picos).
    """
    # Encontrar el índice del pico principal (cerca de la frecuencia de la señal)
    idx_peak = np.argmin(np.abs(freqs - freq))
    
    # Altura del pico principal
    peak_height = amplitud_espectro[idx_peak]
    
    # Crear máscara para excluir el pico principal y sus alrededores
    startt = max(0, idx_peak - window_size)
    endd   = min(len(freqs), idx_peak + window_size + 1)
    
    # Calcular desviación estándar del fondo (parte sin picos)
    background = np.delete(amplitud_espectro, np.arange(startt, endd))
    
    background_std = np.std(background)
    
    # Evitar división por cero
    if background_std == 0:
        return peak_height, 0
    
    SN_freq = peak_height / background_std
    return SN_freq, background_std




# Generar valores de SN_time (logarítmicamente distribuídos de 0.01 a 1.0)
num_datasets = 100 
SN_time_values = np.logspace(-2, 0, num_datasets)  # 0.01 a 1.0  #np.logspace(a, b, N) → genera N valores igualmente espaciados en escala logarítmica, desde 10**𝑎 ahasta 10**b.
#Entonces SN_time_values es un array con 100 valores entre 0.01 y 1.0, pero distribuidos logarítmicamente
# Arrays para almacenar resultados

SN_freq_values = []
background_std_values = []


# Generar muchos conjuntos de datos y calcular SN_freq para cada uno
for SN_time in SN_time_values:
    # Calcular A a partir de SN_time (SN_time = A/noise)
    A = SN_time * noise
    
    # Generar datos
    t, y = generate_data(tmax, dt, A, freq, noise)
    
    # Calcular transformada de Fourier
    F = Fourier_transform(t, y, freqs)
    amplitud_espectro = np.abs(F)
    
    # Calcular SN_freq
    SN_freq, background_std = calculate_SN_freq(amplitud_espectro, freq, freqs)
    
    SN_freq_values.append(SN_freq)
    background_std_values.append(background_std)

# Graficar SN_freq vs SN_time en escala log-log
plt.figure(figsize=(10, 6))
plt.loglog(SN_time_values, SN_freq_values, 'bo-', alpha=0.7, markersize=4)
plt.xlabel('SN_time', fontsize=12)
plt.ylabel('SN_freq', fontsize=12)
plt.title('SN_freq vs SN_time (log-log)', fontsize=14)
plt.grid(True, alpha=0.3, which='both')

# Ajustar modelo lineal en log-log (potencia)
log_SN_time = np.log10(SN_time_values)
log_SN_freq = np.log10(SN_freq_values)   #log(SNfreq​)=m⋅log(SNtime​)+b


valid_indices = ~(np.isinf(log_SN_freq) | np.isnan(log_SN_freq)) #Esta línea crea un filtro para quedarse solo con los valores válidos.



if np.sum(valid_indices) > 2:
    coeffs = np.polyfit(log_SN_time[valid_indices], log_SN_freq[valid_indices], 1)  #np.polyfit(..., 1) ajusta recta (grad1).
    exponent = coeffs[0]
    intercept = coeffs[1]
    
    # Graficar ajuste
    fit_line = 10**(intercept) * SN_time_values**exponent
    plt.loglog(SN_time_values, fit_line, 'r--', 
               label=f'Ajuste: SN_freq ∝ SN_time^{exponent:.2f}')
    plt.legend()

plt.tight_layout()
plt.savefig("1.b.pdf", bbox_inches='tight', dpi=300)
plt.show()



# Función para calcular el ancho del pico
def calculate_peak_width(amplitud_espectro, freqs, threshold=0.5):
    peak_index = np.argmax(amplitud_espectro)
    peak_freq = freqs[peak_index]
    peak_height = amplitud_espectro[peak_index]
    
    # Encontrar el ancho del pico en el umbral especificado
    threshold_height = peak_height * threshold
    left_index = np.where(amplitud_espectro[:peak_index] < threshold_height)[0]
    right_index = np.where(amplitud_espectro[peak_index:] < threshold_height)[0]
    
    if len(left_index) > 0 and len(right_index) > 0:
        width = freqs[right_index[0] + peak_index] - freqs[left_index[-1]]
    else:
        width = 0  # Si no se encuentra el ancho, se establece en 0
    
    return width

# Generar diferentes tmax y calcular el ancho del pico
tmax_values = np.linspace(1, 10, 10)  # Cambiar tmax de 1 a 10
peak_widths = []

for tmax in tmax_values:
    t, y = generate_data(tmax, dt, A, freq, noise)
    F = Fourier_transform(t, y, freqs)
    amplitud_espectro = np.abs(F)
    width = calculate_peak_width(amplitud_espectro, freqs)
    peak_widths.append(width)

# Graficar el ancho del pico en función de tmax
plt.figure(figsize=(10, 6))
plt.plot(tmax_values, peak_widths, 'bo-', markersize=4)
plt.xlabel('tmax', fontsize=12)
plt.ylabel('Ancho del pico', fontsize=12)
plt.title('Ancho del pico en función de tmax', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("1.c.pdf", bbox_inches='tight', dpi=300)
plt.show()
