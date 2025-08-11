import pydicom
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Uso: python dcm_to_png.py input.dcm output.png")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Leer DICOM
ds = pydicom.dcmread(input_path)

# Extraer imagen y normalizar
img = ds.pixel_array
plt.imsave(output_path, img, cmap='gray')
print(f"Imagen guardada en {output_path}")
