import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SEQ_LEN = 5            # tareas previas que ve el modelo
TAREAS_POR_ESTUDIANTE = 20

# =========================
# SECUENCIAS SINTÉTICAS A PARTIR DEL DATASET
# =========================
# No hay historial de estudiantes reales: se agrupan las filas de dataset.csv
# en "estudiantes" simulados y se generan el tiempo real y el cumplimiento.
df = pd.read_csv("dataset.csv")
n_estudiantes = len(df) // TAREAS_POR_ESTUDIANTE
df = df.iloc[:n_estudiantes * TAREAS_POR_ESTUDIANTE].copy()
df["estudiante"] = np.repeat(np.arange(n_estudiantes), TAREAS_POR_ESTUDIANTE)

# ritmo propio de cada estudiante (>1 = tarda más de lo estimado)
ritmo = np.random.uniform(0.7, 1.5, n_estudiantes)
ritmo_fila = ritmo[df["estudiante"].values]

ruido = np.random.normal(1.0, 0.15, len(df)).clip(0.5, 1.5)
df["tiempo_real"] = df["estimated_time"] * (1 + df["dificultad"] / 20) * ritmo_fila * ruido
df["cumplimiento"] = (df["tiempo_real"] <= df["estimated_time"] * 1.2).astype(float)

# las horas recomendadas se ajustan al ritmo del estudiante,
# así el historial aporta información que el modelo puede aprender
df["horas_recomendadas"] = np.minimum(
    np.maximum(df["horas_recomendadas"] * ritmo_fila, 0.5), df["horas"]
)

feat_hist = ["dificultad", "estimated_time", "tiempo_real", "cumplimiento"]
feat_actual = ["horas", "dificultad", "estimated_time", "dias", "tasks"]

scaler_hist = MinMaxScaler().fit(df[feat_hist])
scaler_actual = MinMaxScaler().fit(df[feat_actual])
H = scaler_hist.transform(df[feat_hist])
A = scaler_actual.transform(df[feat_actual])

X_seq, X_act, y_p, y_h, grupo = [], [], [], [], []
for est in range(n_estudiantes):
    ini = est * TAREAS_POR_ESTUDIANTE
    for t in range(SEQ_LEN, TAREAS_POR_ESTUDIANTE):
        i = ini + t
        X_seq.append(H[i - SEQ_LEN:i])
        X_act.append(A[i])
        y_p.append(df["prioridad"].iloc[i])
        y_h.append(df["horas_recomendadas"].iloc[i])
        grupo.append(est)

X_seq, X_act = np.array(X_seq), np.array(X_act)
y_p, y_h, grupo = np.array(y_p), np.array(y_h), np.array(grupo)

# separar por estudiante para que validación tenga estudiantes no vistos
est_val = np.random.permutation(n_estudiantes)[: int(n_estudiantes * 0.2)]
val = np.isin(grupo, est_val)
tr = ~val

# =========================
# MODELO LSTM
# =========================
seq_input = keras.Input(shape=(SEQ_LEN, len(feat_hist)), name="historial")
act_input = keras.Input(shape=(len(feat_actual),), name="tarea_actual")

h = keras.layers.LSTM(32)(seq_input)
x = keras.layers.Concatenate()([h, act_input])
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dense(32, activation="relu")(x)

prioridad_output = keras.layers.Dense(3, activation="softmax", name="prioridad")(x)
horas_output = keras.layers.Dense(1, activation="softplus", name="horas")(x)

model = keras.Model(
    [seq_input, act_input], {"prioridad": prioridad_output, "horas": horas_output}
)
model.compile(
    optimizer="adam",
    loss={"prioridad": "sparse_categorical_crossentropy", "horas": "mse"},
    loss_weights={"prioridad": 1.0, "horas": 0.7},
    metrics={"prioridad": "accuracy", "horas": "mae"},
)

history = model.fit(
    [X_seq[tr], X_act[tr]],
    {"prioridad": y_p[tr], "horas": y_h[tr]},
    validation_data=([X_seq[val], X_act[val]], {"prioridad": y_p[val], "horas": y_h[val]}),
    epochs=60,
    batch_size=32,
    verbose=2,
)

model.save("modelo_lstm.h5")

# =========================
# GRÁFICAS DE ENTRENAMIENTO
# =========================
hist = history.history
epocas = np.arange(1, len(hist["loss"]) + 1)
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"

paneles = [
    ("prioridad_loss", "Pérdida – prioridad", "Entropía cruzada"),
    ("prioridad_accuracy", "Precisión – prioridad", "Accuracy"),
    ("horas_loss", "Pérdida – horas recomendadas", "MSE"),
    ("horas_mae", "Error absoluto – horas recomendadas", "MAE (horas)"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (clave, titulo, ylabel) in zip(axes.flat, paneles):
    ax.plot(epocas, hist[clave], color=C_TRAIN, lw=2, label="Entrenamiento")
    ax.plot(epocas, hist["val_" + clave], color=C_VAL, lw=2, label="Validación")
    ax.set_title(titulo, fontsize=11, loc="left", color="#222")
    ax.set_xlabel("Época", color="#555")
    ax.set_ylabel(ylabel, color="#555")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors="#555")
    # etiqueta solo el valor final de validación
    ax.annotate(f"{hist['val_' + clave][-1]:.3f}", (epocas[-1], hist["val_" + clave][-1]),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color="#333")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
fig.suptitle("Entrenamiento del modelo LSTM (secuencias de 5 tareas previas)",
             x=0.02, ha="left", fontsize=13, color="#222")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("grafica_entrenamiento_lstm.png", dpi=200)
print("Gráfica guardada en grafica_entrenamiento_lstm.png")
