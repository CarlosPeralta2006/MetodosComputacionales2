import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d


data = np.loadtxt("Data2/tomography_data/OGLE-LMC-CEP-0001.dat")
t = data[:, 0]    # tiempo en días
mag = data[:, 1]  # brillo (magnitud)
err = data[:, 2]  # incertidumbre


fmin = 0.01
fmax = 10
frequency = np.linspace(fmin, fmax, 10000)

ls = LombScargle(t, mag, err)
power = ls.power(frequency)

best_frequency_ls = frequency[np.argmax(power)]
print("Frecuencia Lomb-Scargle:", best_frequency_ls, "ciclos/día")

phi_ls = np.mod(best_frequency_ls * t, 1)

plt.figure(figsize=(8,5))
plt.errorbar(phi_ls, mag, yerr=err, fmt='o', markersize=3, color='blue', ecolor='gray', alpha=0.7)
plt.gca().invert_yaxis()
plt.xlabel("Fase")
plt.ylabel("Brillo (mag)")
plt.title("Curva de fase (Lomb-Scargle)")
plt.savefig("4.pdf")  # <- aq
plt.grid(alpha=0.3)
plt.show()


dt = 1.0  
t_uniform = np.arange(t.min(), t.max(), dt)
interp_func = interp1d(t, mag, kind='linear')
mag_uniform = interp_func(t_uniform)


N = len(t_uniform)
fft_vals = np.fft.fft(mag_uniform - np.mean(mag_uniform))
fft_freq = np.fft.fftfreq(N, d=dt)


pos_mask = fft_freq > 0
fft_vals = np.abs(fft_vals[pos_mask])
fft_freq = fft_freq[pos_mask]

best_frequency_fft = fft_freq[np.argmax(fft_vals)]
print("Frecuencia FFT:", best_frequency_fft, "ciclos/día")

phi_fft = np.mod(best_frequency_fft * t, 1)


phi_double = np.concatenate([phi_fft, phi_fft+1])
mag_double = np.concatenate([mag, mag])
err_double = np.concatenate([err, err])

plt.figure(figsize=(8,5))
plt.errorbar(phi_double, mag_double, yerr=err_double, fmt='o', markersize=3, alpha=0.7)
plt.gca().invert_yaxis()
plt.xlabel("Fase")
plt.ylabel("Brillo (mag)")
plt.title(f"Curva de fase (FFT), f = {best_frequency_fft:.4f} ciclos/día")
plt.savefig("4.1.pdf") 
plt.grid(alpha=0.3)
plt.show()
