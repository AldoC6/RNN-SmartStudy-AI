import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

data = []

for _ in range(1000): 
    horas = np.random.uniform(1, 12)  # horas disponibles reales
    dificultad = np.random.randint(1, 11)
    estimated_time = np.random.uniform(0.1, 5)  # ahora incluye casos pequeños
    dias = np.random.randint(1, 10)
    tasks = np.random.randint(1, 6)

    # corregir inconsistencias (caso tipo: dificultad alta + tiempo muy bajo)
    if dificultad > 8 and estimated_time < 0.5:
        estimated_time *= 1.5

    # urgencia mejor definida
    urgencia = 1 / (dias + 0.1)

    score = urgencia * 0.5 + (dificultad / 10) * 0.3 + (tasks / 5) * 0.2

    if score > 0.65:
        prioridad = 2  # alta
    elif score > 0.35:
        prioridad = 1  # media
    else:
        prioridad = 0  # baja

    # horas recomendadas con límite real (CLAVE)
    horas_por_dia = (estimated_time / dias) * (1 + dificultad / 10)
    horas_por_dia = min(horas_por_dia, horas)

    data.append([
        horas, dificultad, estimated_time, dias, tasks,
        prioridad, horas_por_dia
    ])

columns = [
    "horas", "dificultad", "estimated_time", "dias", "tasks",
    "prioridad", "horas_recomendadas"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)

print(df.head())

# =========================
# PREPARAR DATOS
# =========================
X = df[["horas", "dificultad", "estimated_time", "dias", "tasks"]]
y_prioridad = df["prioridad"]
y_horas = df["horas_recomendadas"]

X_train, X_test, y_p_train, y_p_test, y_h_train, y_h_test = train_test_split(
    X, y_prioridad, y_horas, test_size=0.2
)


input_layer = keras.Input(shape=(5,))

x = keras.layers.Dense(32, activation='relu')(input_layer)
x = keras.layers.Dense(32, activation='relu')(x)

# 🔥 salida clasificación
prioridad_output = keras.layers.Dense(
    3,
    activation='softmax',
    name='prioridad'
)(x)

# 🔥 salida regresión
horas_output = keras.layers.Dense(
    1,
    activation='relu',
    name='horas'
)(x)

model = keras.Model(
    inputs=input_layer,
    outputs=[prioridad_output, horas_output]
)

# =========================
# COMPILAR
# =========================
model.compile(
    optimizer='adam',
    loss={
        'prioridad': 'sparse_categorical_crossentropy',
        'horas': 'mse'
    },
    metrics={
        'prioridad': 'accuracy',
        'horas': 'mae'
    }
)


model.fit(
    X_train,
    {
        'prioridad': y_p_train,
        'horas': y_h_train
    },
    epochs=50,
    validation_data=(
        X_test,
        {
            'prioridad': y_p_test,
            'horas': y_h_test
        }
    )
)

model.save("modelo_estudio.h5")
