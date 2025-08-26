
import numpy as np
import matplotlib.pyplot as plt




def generate_data(tmax, dt, A, freq, noise, sampling_noise=0):
   
    ts = np.arange(0, tmax + dt, dt)
    
    # Agregar ruido en el tiempo de muestreo si se indica
    if sampling_noise > 0:
        ts = ts + np.random.normal(0, sampling_noise, size=ts.shape)
        ts = np.sort(ts)  # ordenar los tiempos para evitar inversión temporal
    
    y = np.random.normal(loc=A * np.sin(2 * np.pi * freq * ts), scale=noise)
    
    return ts, y




def Fourier_transform(t, y, freqs):

    N = len(t)
    F = np.zeros(len(freqs), dtype=complex)  # arreglo complejo para la transformada

   #Reserva un vector complejo F de longitud igual al número de frecuencias.Cada uno almancenando la transformada en la frecuencia freqs[k].
   
    for k, fk in enumerate(freqs):   
        expo = np.exp(-2j * np.pi * fk * t)  # e^{-2πi f t} es un vector (porque t es un array).
        F[k] = np.sum(y * expo)

    return F



# Parámetros base
tmax = 3
dt = 0.01
A = 1
freq = 5
noise = 0.2
f_nyq = 1 / (2*dt)
freqs = np.linspace(0, 2.7*f_nyq, 2000)

# Diferentes valores de ruido de muestreo
sampling_noises = [0, 0.001, 0.005, 0.01]

plt.figure(figsize=(12, 6))

for sn in sampling_noises:
    t, y = generate_data(tmax, dt, A, freq, noise, sampling_noise=sn)
    F = Fourier_transform(t, y, freqs)
    amplitud_espectro = np.abs(F)
    plt.plot(freqs, amplitud_espectro, label=f'sampling_noise={sn}')

plt.axvline(x=freq, color='red', linestyle='--', alpha=0.8, label=f'Freq real {freq} Hz')
plt.axvline(x=f_nyq, color='green', linestyle='--', alpha=0.8, label=f'Nyquist {f_nyq:.1f} Hz')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud del espectro |F(f)|")
plt.title("Transformada de Fourier con ruido en el muestreo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2.7*f_nyq)
plt.tight_layout()
plt.savefig("1.d.pdf", bbox_inches='tight', dpi=300)
plt.show()
