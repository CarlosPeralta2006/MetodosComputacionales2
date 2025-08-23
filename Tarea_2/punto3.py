import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# - - - 3.a - - - - - - - - - 

# se trabajará en el espacio de Fourier
img = np.array(Image.open("Data2/tomography_data/miette.jpg"))
Y, X, C = img.shape  # donde C son los canales de color

img_filt = np.zeros_like(img, dtype=float)

freqx = np.fft.fftfreq(X, d=1)
freqy = np.fft.fftfreq(Y, d=1)
# construcción de Gaussiana
FX, FY = np.meshgrid(np.fft.fftshift(freqx), np.fft.fftshift(freqy))
sigma = 0.025  # parámetro para la gaussiana (más cercano a 0, más borroso)
gaussian2D = np.exp(-0.5 * ((FX**2 + FY**2)/sigma**2))

for c in range(C):   # realizar la trasnformada para cada uno de los 3 canales (R,G,B)
    F = np.fft.fft2(img[:, :, c])   
    F_shift = np.fft.fftshift(F)           
    F_filt = F_shift * gaussian2D            
    canal_filt = np.fft.ifft2(np.fft.ifftshift(F_filt)).real
    img_filt[:, :, c] = canal_filt  

plt.figure(figsize=(12,12))
plt.imshow(np.clip(img_filt, 0, 255).astype(np.uint8))  
plt.axis("off")                   
plt.savefig("3.a.pdf", bbox_inches="tight", pad_inches=0.)


# - - - 3.b - - - - - - - - - 

pato = np.array(Image.open('Data2/tomography_data/p_a_t_o.jpg'))
yp, xp = pato.shape
print(pato.shape)
X,Y = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))


plt.figure(figsize=(12,12))
plt.imshow(np.clip(img_filt, 0, 255).astype(np.uint8))  
plt.axis("off")                   
plt.savefig("3.b.a.pdf", bbox_inches="tight", pad_inches=0.)
