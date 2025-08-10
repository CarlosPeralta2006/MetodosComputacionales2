import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from Punto1 import data

# --- helper ---
def mad_std(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return mad / 0.6745 if mad != 0 else np.std(a)

def estimate_baseline(y, win_frac=0.06, smooth_sigma=6):
    n = len(y)
    win = int(max(3, round(win_frac * n)))
    if win % 2 == 0: win += 1
    med = pd.Series(y).rolling(window=win, center=True, min_periods=1).median().values
    return gaussian_filter1d(med, sigma=smooth_sigma)

def detect_edges_outside_belly(df,
                               baseline_win_frac=0.06, baseline_smooth_sigma=6,
                               peak_height_rel=0.03, min_snr=6.0, rel_height=0.5):
    """
    Devuelve índices left/right (bordes) de picos significativos fuera de la barriga principal.
    Si no hay picos significativos fuera de la barriga, devuelve array vacío.
    """
    x = df['Energía'].values
    y = df['Fotones'].values
    dx = np.mean(np.diff(x))

    # 1) Baseline y residuo
    baseline = estimate_baseline(y, win_frac=baseline_win_frac, smooth_sigma=baseline_smooth_sigma)
    residual = y - baseline
    residual[residual < 0] = 0.0

    if residual.max() <= 0:
        return np.array([], dtype=int)

    # 2) ruido y umbral combinado
    sigma_noise = mad_std(residual)
    height_thresh = max(peak_height_rel * residual.max(), min_snr * sigma_noise)

    # 3) detectar picos en residual
    peaks, props = find_peaks(residual, height=height_thresh, prominence=0.5*height_thresh, distance=2)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # 4) obtener left/right (en índices) de cada pico (FWHM)
    widths, left_ips, right_ips, _ = peak_widths(residual, peaks, rel_height=rel_height)
    left_idx = np.rint(left_ips).astype(int)
    right_idx = np.rint(right_ips).astype(int)

    # 5) identificar la región de la barriga principal en y (zona del máximo global)
    main_idx = int(np.argmax(y))
    # punto de corte para definir la barriga principal: por ejemplo 50% del máximo global de y
    belly_thresh = 0.5 * y.max()
    above = np.where(y >= belly_thresh)[0]
    if len(above) == 0:
        main_left, main_right = -1, -1
    else:
        main_left, main_right = int(above[0]), int(above[-1])

    # 6) quedarnos solo con picos cuyos left-right no solapen la barriga principal
    kept_edges = []
    for li, ri in zip(left_idx, right_idx):
        # si el pico está completamente fuera de la barriga principal, lo guardamos
        if ri < main_left or li > main_right:
            kept_edges.append(li)
            kept_edges.append(ri)

    return np.unique(np.array(kept_edges, dtype=int))

# --- uso con tus kV ---
Mo, Rh, W = data()

kV1 = '27kV'
kV2 = '10kV'
kV3 = '38kV'

cases = [('Molibdeno (Mo)', Mo[kV1], 'b', kV1),
         ('Rodio (Rh)', Rh[kV2], 'r', kV2),
         ('Tungsteno (W)', W[kV3], 'g', kV3)]

fig, axes = plt.subplots(1, 3, figsize=(18,6))
plt.subplots_adjust(wspace=0.3)

for ax, (name, df, color, kv) in zip(axes, cases):
    edges_idx = detect_edges_outside_belly(df,
                                          baseline_win_frac=0.06,
                                          baseline_smooth_sigma=6,
                                          peak_height_rel=0.03,   # 3% del residual máximo
                                          min_snr=6.0,
                                          rel_height=0.5)
    ax.plot(df['Energía'], df['Fotones'], color=color, linewidth=1.4)
    if len(edges_idx) > 0:
        ax.plot(df['Energía'].iloc[edges_idx], df['Fotones'].iloc[edges_idx], 'ro', ms=6)
    # debug: marcar la barriga principal (opcional, comentar si no quieres)
    # main = np.where(df['Fotones'] >= 0.5*df['Fotones'].max())[0]
    # if len(main)>0:
    #     ax.plot(df['Energía'].iloc[main[0]:main[-1]+1], df['Fotones'].iloc[main[0]:main[-1]+1], color='0.6', alpha=0.2)

    ax.set_title(f"{name} a {kv}")
    ax.set_xlim(0,35)
    ax.set_xlabel("Energía (keV)")
    ax.grid(ls='--', alpha=0.5)

axes[0].set_ylabel("Conteo de fotones (u.a.)")
plt.tight_layout()
plt.show()
