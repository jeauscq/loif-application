import keras
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

train_subset_size = 60000
test_subset_size = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Crear un DataFrame vacío para almacenar las métricas
data = {
    'Index 1': [],
    'Index 2': [],
    'Mean Absolute Error (MAE)': [],
    'Mean Squared Error (MSE)': [],
    'Structural Similarity Index (SSIM)': []
}
df = pd.DataFrame(data)
df_list = []
# I need to develop an algorithm that takes a n amount of pair of images an calculates the similarity between them an creates a subdataset in which every entry is a pair of images with a similarity associated

for i in range(int(train_subset_size/2)):
    index_1 = i
    index_2 = train_subset_size - (i+1)

    image1 = x_train[index_1]
    image2 = x_train[index_2]

    # Calcular el SSIM entre las dos imágenes
    ssim_index, _ = ssim(image1, image2, full=True)
    # Calcular el Mean Absolute Error (MAE)
    mae = np.mean(np.abs(image1 - image2))
    # Calcular el Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Crear un DataFrame para esta iteración
    df_temp = pd.DataFrame([{
        'Index 1': index_1,
        'Index 2': index_2,
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Structural Similarity Index (SSIM)': ssim_index
    }])

    # Agregar el DataFrame temporal a la lista
    df_list.append(df_temp)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(df_list, ignore_index=True)

# Calcular el promedio de las métricas por fila
df['Average'] = df[['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Structural Similarity Index (SSIM)']].mean(axis=1)


# Normalizar cada columna de 0 a 1
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
# Aplicar umbral de 0.5 y redondear la columna 'Average'
df_normalized['Average'] = (df_normalized['Average'] > 0.5).astype(int)

# Guardar el DataFrame normalizado en un archivo Excel
df_normalized.to_excel('train_image_comparison_normalized.xlsx', index=False)

df = pd.DataFrame(data)
df_list = []
# I need to develop an algorithm that takes a n amount of pair of images an calculates the similarity between them an creates a subdataset in which every entry is a pair of images with a similarity associated

for i in range(int(test_subset_size/2)):
    index_1 = i
    index_2 = test_subset_size - (i+1)

    image1 = x_test[index_1]
    image2 = x_test[index_2]

    # Calcular el SSIM entre las dos imágenes
    ssim_index, _ = ssim(image1, image2, full=True)
    # Calcular el Mean Absolute Error (MAE)
    mae = np.mean(np.abs(image1 - image2))
    # Calcular el Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Crear un DataFrame para esta iteración
    df_temp = pd.DataFrame([{
        'Index 1': index_1,
        'Index 2': index_2,
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Structural Similarity Index (SSIM)': ssim_index
    }])

    # Agregar el DataFrame temporal a la lista
    df_list.append(df_temp)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(df_list, ignore_index=True)

# Calcular el promedio de las métricas por fila
df['Average'] = df[['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Structural Similarity Index (SSIM)']].mean(axis=1)

# Normalizar cada columna de 0 a 1
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
# Aplicar umbral de 0.5 y redondear la columna 'Average'
df_normalized['Average'] = (df_normalized['Average'] > 0.5).astype(int)

# Guardar el DataFrame normalizado en un archivo Excel
df_normalized.to_excel('test_image_comparison_normalized.xlsx', index=False)
