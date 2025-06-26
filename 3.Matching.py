import numpy as np

contador = 0

for i in range(100):
    result = np.load(f'distances/90/results{i+1}.npy')
    if i == np.argmin(result):
        contador += 1

print(f'Hubo {contador} coindidencias')