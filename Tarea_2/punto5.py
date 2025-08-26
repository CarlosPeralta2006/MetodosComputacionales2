import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# 1. Cargar las proyecciones
proyecciones = np.load("Data2/tomography_data/6.npy")
num_angulos, num_pixeles = proyecciones.shape
angulos = np.linspace(0, 180, num_angulos, endpoint=False)

def filtro_pasa_altas(signal):
    N = len(signal)
    # Transformada de Fourier
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N).reshape(-1, 1)
    # Filtro tipo rampa: |freq|
    ramp = np.abs(freqs)
    # Aplicar filtro
    fft_filtrado = fft * ramp.flatten()
    # Regresar al dominio temporal
    return np.real(np.fft.ifft(fft_filtrado))

# 3. Reconstrucción
rows = num_pixeles
reconstruccion = np.zeros((rows, rows))

for i, ang in enumerate(angulos):
    # Señal original
    signal = proyecciones[i]
    # Filtrar señal
    filtrada = filtro_pasa_altas(signal)
    # Expandir a 2D (repetir en filas)
    proy_2D = np.tile(filtrada[:, None], (1, rows)).T
    # Rotar
    img_rotada = ndi.rotate(proy_2D, ang, reshape=False, mode="reflect")
    # Acumular
    reconstruccion += img_rotada

# 4. Guardar y mostrar
plt.imshow(reconstruccion, cmap="inferno")
plt.title("Reconstrucción Tomográfica Filtrada")
plt.axis("off")
plt.savefig("4.png", dpi=300)
