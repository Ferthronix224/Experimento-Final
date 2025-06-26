import numpy as np
import cv2

for ii in range(1, 101):
    print(f'Imagen {ii}')
    desc_1 = np.load(f'descriptors/90/des_o{ii}.npy')
    distances_l = list()
    for i in range(2, 101):
        desc_2 = np.load(f'descriptors/90/des_t{i}.npy')
        
        # Dos conjuntos de descriptores (matrices)
        # A: descriptores de imagen 1 (NxD)
        # B: descriptores de imagen 2 (MxD)
        A = desc_1.astype(np.float32)
        B = desc_2.astype(np.float32)

        # Configurar FLANN
        index_params = dict(algorithm=1, trees=5)  # algoritmo 1 = KDTree
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Usar knnMatch para obtener los mejores emparejamientos
        matches = flann.knnMatch(A, B, k=1)  # Empareja cada vector de A con el más cercano en B
        del flann

        # Extraer distancias
        distances = [m[0].distance for m in matches]
        del matches

        # Distancia promedio o mínima
        mean_distance = np.mean(distances)
        distances_l.append(mean_distance)
        # min_distance = np.min(distances)

        # print(f"Distancia promedio {i}:", mean_distance)
        # print("Distancia mínima:", min_distance)

    np.save(f'distances/90/results{ii}.npy', distances_l)