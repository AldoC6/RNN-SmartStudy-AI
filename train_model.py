import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# generar datos
data = []

for _ in range(500):
    horas = np.random.uniform(2, 12)  # horas disponibles
    dificultad = np.random.randint(1, 11)
    estimated_time = np.random.uniform(0.5, 5)  # horas (ya convertidas)
    dias = np.random.randint(1, 10)
    tasks = np.random.randint(1, 6)

    # lógica base (simple, NO la compliques más)
    urgencia = 1 / dias
    score = urgencia * 0.5 + (dificultad / 10) * 0.3 + (tasks / 5) * 0.2

    if score > 0.6:
        prioridad = 2
    elif score > 0.3:
        prioridad = 1
    else:
        prioridad = 0

    horas_por_dia = (estimated_time / dias) * (1 + dificultad / 10)

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


X = df[["horas", "dificultad", "estimated_time", "dias", "tasks"]]
y = df[["prioridad", "horas_recomendadas"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2)  # prioridad + horas
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save("modelo_estudio.h5")