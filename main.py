from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import numpy as np


app = FastAPI()

model = keras.models.load_model("modelo_estudio.h5", compile=False)

# Modelo de datos que recibirá la API
class InputData(BaseModel):
    horas: float
    dificultad: int
    dias_restantes: int
    estimated_time: float
    tasks: int

@app.get("/")
def home():
    return {"mensaje": "API funcionando"}

@app.post("/predictLogic")
def predict(data: InputData):
    # lógica simple (simulación de IA)

    dias = data.dias_restantes if data.dias_restantes > 0 else 1
    # horas necesarias totales
    horas_totales = data.estimated_time

    # distribución base
    horas_por_dia = horas_totales / dias

    # ajuste por dificultad (más dificultad = más tiempo diario)
    factor_dificultad = 1 + (data.dificultad / 10)

    horas_recomendadas = horas_por_dia * factor_dificultad
    
    prioridad = 0
    # prioridad lógica
    if data.dias_restantes <= 2:
        prioridad = 2
    elif data.dias_restantes <= 5:
        prioridad = 1
    else:
        prioridad = 0

    return {
        "prioridad": prioridad,
        "horas_recomendadas": round(horas_recomendadas, 2)
    }


@app.post("/predictRN")
def predict(data: InputData):

    entrada = np.array([[
        data.horas,
        data.dificultad,
        data.estimated_time,
        data.dias_restantes,
        data.tasks
    ]])

    pred_prioridad, pred_horas = model.predict(entrada)

    # 🔥 prioridad = índice con mayor probabilidad
    prioridad = int(np.argmax(pred_prioridad[0]))

    # 🔥 horas (ya es número)
    horas_recomendadas = float(pred_horas[0][0])

    return {
        "prioridad": prioridad,
        "horas_recomendadas": round(horas_recomendadas, 2)
    }