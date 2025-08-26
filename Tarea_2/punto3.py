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

plt.rcParams["image.cmap"] = "gray"

pato = np.array(Image.open('Data2/tomography_data/p_a_t_o.jpg'))
yp, xp = pato.shape
X,Y = np.meshgrid(np.arange(pato.shape[1]),np.arange(pato.shape[0]))

F = np.fft.fft2(pato)
F = np.fft.fftshift(F)

# cuadros a eliminar del espacio de frecuencias - tanteo
Z = (abs(X-256) < 7.5) & (abs(Y-128) < 121) 
W = (abs(X-256) < 7.5) & (abs(Y-384) < 121) 
H = (abs(X-128) < 121) & (abs(Y-256) < 7)
E = (abs(X-384) < 121) & (abs(Y-256) < 7)
Z_line = np.abs(Y - (-0.34*X + 340)) < 12
Z_line2 = np.abs(Y - (X)) < 10
Z_ring = (np.hypot(X-256,Y-256) <= 13.5) & (np.hypot(X-256,Y-256) >= 4)

let = np.hypot(X-256,Y-256) <= 4.4

F_del = (1-0.95*Z) * (1-0.95*W) * (1-H) * (1-E) * (1-0.95*Z_line) * (1-0.7*Z_ring) * (1-0.8*Z_line2)
F_fil = np.maximum(F_del, let.astype(float))
F_filt = F * F_fil

pato_filt = np.fft.ifft2(np.fft.ifftshift(F_filt))

plt.figure(figsize=(12,12))
plt.imshow(pato_filt.real)  
plt.axis("off")                   
plt.savefig("3.b.a.jpg", bbox_inches="tight", pad_inches=0.)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

gato = np.array(Image.open('Data2/tomography_data/g_a_t_o.png'))
yg, xg = gato.shape
x, y = np.meshgrid(np.arange(gato.shape[1]),np.arange(gato.shape[0]))

Ga = np.fft.fft2(gato)
Ga = np.fft.fftshift(Ga)

#cuadros a eliminar del espacio de frecuencias - tanteo
Z_linea1 = np.abs(y - (2*x - 370)) < 42  # principal fuente de interferencia
Z_linea2 = np.abs(y - (-2*x + 1120)) < 5
Z_linea3 = np.abs(y - (375)) < 2.1
Z_ring1 = (np.hypot(x-375.5,y-377) <= 23.5) & (np.hypot(x-375.5,y-377) >= 4.85)

let1 = np.hypot(x-375.5,y-377) <= 9.5

G_del = (1-0.98*Z_linea1) * (1-0.95*Z_linea2)  * (1-0.75*Z_linea3) * (1-0.96*Z_ring1)
G_fil = np.maximum(G_del, let1.astype(float))
G_filt = Ga * G_fil

gato_filt = np.fft.ifft2(np.fft.ifftshift(G_filt))

plt.figure(figsize=(12,12))
plt.imshow(gato_filt.real)  
plt.axis("off")                   
plt.savefig("3.b.b.png", bbox_inches="tight", pad_inches=0.)