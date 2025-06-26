import numpy as np
from skimage import io, color
from skimage.feature import hog
from skimage.util import img_as_float
from skimage.transform import resize

def extract_hog_descriptors(image, keypoints, patch_size=32, hog_pixels_per_cell=(8, 8), orientations=9, cells_per_block=(2, 2)):
    """
    Extrae descriptores HOG a partir de coordenadas de keypoints en la imagen.

    image: ndarray (HxWx3 o HxW) - imagen original
    keypoints: lista de (x, y) - coordenadas de puntos clave
    patch_size: tamaño de la ventana alrededor de cada punto (en píxeles)
    returns: lista de vectores HOG
    """
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = img_as_float(image)

    descriptors = []

    half_size = patch_size // 2
    for (x, y) in keypoints:
        # Extraer parche centrado en el keypoint
        x, y = int(x), int(y)
        xmin = max(x - half_size, 0)
        xmax = min(x + half_size, image.shape[1])
        ymin = max(y - half_size, 0)
        ymax = min(y + half_size, image.shape[0])

        patch = image[ymin:ymax, xmin:xmax]

        # Asegurar que el parche tenga el tamaño correcto
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = resize(patch, (patch_size, patch_size), anti_aliasing=True)

        # Calcular descriptor HOG
        hog_descriptor = hog(
            patch,
            orientations=orientations,
            pixels_per_cell=hog_pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )
        descriptors.append(hog_descriptor)

    return descriptors

# Imagen y keypoints de ejemplo
for i in range(1, 101):
# i = 2
    image = io.imread(f'img/originals/{i}.jpg')
    keypoints = np.load(f'keypoints/90/original{i}.npy')

    hog_descs = extract_hog_descriptors(image, keypoints)
    hog_descs = np.array(hog_descs)

    np.save(f'descriptors/90/des_o{i}', hog_descs)