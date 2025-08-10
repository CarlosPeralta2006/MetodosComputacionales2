import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from Punto1 import data

# Función para detectar centros de picos característicos
def picos_fwhm(df, baseline_sigma=20, max_fwhm_keV=0.8,
               peak_height_factor=0.05, prominence_factor=0.05):
    x = df['Energía'].values
    y = df['Fotones'].values
    dx = np.mean(np.diff(x))  # keV por muestra

    # Suavizado para estimar el baseline
    baseline = gaussian_filter1d(y, sigma=baseline_sigma)
    residual = y - baseline
    residual[residual < 0] = 0

    # Detección de picos sobre el residual
    peaks, _ = find_peaks(residual,
                          height=peak_height_factor * residual.max(),
                          prominence=prominence_factor * residual.max())

    # Filtrar por FWHM
    widths = peak_widths(residual, peaks, rel_height=0.5)[0] * dx
    keep = widths <= max_fwhm_keV
    peaks = peaks[keep]

    return peaks

# Datos del taller
spectra = data()
Mo, Rh, W = spectra

# Lista de todos los kV disponibles
kv_list = list(Mo.keys())

# Colores para cada material
colores = {'Mo': 'b', 'Rh': 'r', 'W': 'g'}

for kV in kv_list:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    materiales = [('Molibdeno (Mo)', Mo[kV], 'Mo'),
                  ('Rodio (Rh)', Rh[kV], 'Rh'),
                  ('Tungsteno (W)', W[kV], 'W')]

    for ax, (label, df, mat) in zip(axes, materiales):
        peaks_idx = picos_fwhm(df)
        ax.plot(df['Energía'], df['Fotones'], color=colores[mat])
        ax.plot(df['Energía'].iloc[peaks_idx], df['Fotones'].iloc[peaks_idx],
                'ro', ms=4)
        ax.set_title(f"{label} a {kV}")
        ax.set_xlabel("Energía (keV)")
        ax.set_xlim(0, 35)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].set_ylabel("Conteo de fotones (u.a.)")

    plt.tight_layout()
    plt.savefig(f"picos_{kV}.pdf")
    plt.close(fig)

print("Figuras guardadas en PDF para todos los kV.")
