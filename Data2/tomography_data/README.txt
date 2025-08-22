Estos datos de tomografía son cortes de lo datos de ejemplo encontrados en el libro "Neural Data Science in Python" (https://neuraldatascience.io).

Cada archivo de datos .npy es una matriz cuyas filas son los datos de proyección para cierto ángulo.
Las proyecciones se realizaron cada 0.5 grados, de 0 a 180, es decir `np.arange(0,180,0.5)`. 

Puede importar los archivos .npy usando `np.load(filename)`

Si alguien en su grupo no está dispuesto a mirar imágenes médicas, pueden usar a skelly.npy. 

También incluí la imagen de muestra de la tarea, por si quieren probar reproducir las imágenes del enunciado.
