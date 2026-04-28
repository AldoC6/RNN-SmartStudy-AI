import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = []

for _ in range(2000):  # más datos = mejor aprendizaje
    horas = np.random.uniform(1, 12)
    dificultad = np.random.randint(1, 11)
    estimated_time = np.random.uniform(0.1, 5)
    dias = np.random.randint(1, 15)  # más rango
    tasks = np.random.randint(1, 6)

    # 🔥 corregir incoherencias
    if dificultad > 8 and estimated_time < 0.5:
        estimated_time *= 2

    # =========================
    # NUEVA LÓGICA INTELIGENTE
    # =========================

    urgencia = 1 / (dias + 1)
    importancia = dificultad / 10
    carga = estimated_time / (horas + 0.1)

    score = (0.5 * urgencia) + (0.3 * importancia) + (0.2 * carga)

    if score > 0.65:
        prioridad = 2
    elif score > 0.35:
        prioridad = 1
    else:
        prioridad = 0

    # 🔥 DISTRIBUCIÓN CORRECTA (CLAVE)
    horas_por_dia = estimated_time / dias

    # ajustar por dificultad (más difícil → más tiempo diario)
    horas_por_dia *= (1 + dificultad / 10)

    # 🔥 mínimo garantizado (EVITA 0)
    horas_por_dia = max(horas_por_dia, 0.5)

    # 🔥 no exceder horas disponibles
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

# =========================
# NORMALIZACIÓN (MUY IMPORTANTE)
# =========================
scaler = MinMaxScaler()
X = df[["horas", "dificultad", "estimated_time", "dias", "tasks"]]
X_scaled = scaler.fit_transform(X)

y_prioridad = df["prioridad"]
y_horas = df["horas_recomendadas"]

X_train, X_test, y_p_train, y_p_test, y_h_train, y_h_test = train_test_split(
    X_scaled, y_prioridad, y_horas, test_size=0.2
)

# =========================
# MODELO
# =========================
input_layer = keras.Input(shape=(5,))

x = keras.layers.Dense(64, activation='relu')(input_layer)
x = keras.layers.Dense(64, activation='relu')(x)

prioridad_output = keras.layers.Dense(
    3,
    activation='softmax',
    name='prioridad'
)(x)

horas_output = keras.layers.Dense(
    1,
    activation='softplus',
    name='horas'
)(x)

model = keras.Model(
    inputs=input_layer,
    outputs=[prioridad_output, horas_output]
)

model.compile(
    optimizer='adam',
    loss={
        'prioridad': 'sparse_categorical_crossentropy',
        'horas': 'mse'
    },
    loss_weights={  # 🔥 balancear importancia
        'prioridad': 1.0,
        'horas': 0.7
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
    epochs=60,
    validation_data=(
        X_test,
        {
            'prioridad': y_p_test,
            'horas': y_h_test
        }
    )
)

model.save("modelo_estudio.h5")