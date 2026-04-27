# 🧠 Red Neuronal API - Smart Study

API desarrollada con FastAPI que utiliza una red neuronal (TensorFlow/Keras) para analizar tareas académicas y generar recomendaciones inteligentes.

## 🚀 Tecnologías utilizadas

* Python
* FastAPI
* TensorFlow / Keras
* NumPy

---

## ⚙️ Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/red_neuronal_api.git
cd red_neuronal_api
```

2. Crear entorno virtual:

```bash
python -m venv venv
```

3. Activar entorno:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

4. Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar la API

```bash
uvicorn main:app --reload
```

La API estará disponible en:

```
http://127.0.0.1:8000
```

---

## 📡 Endpoint principal

### POST `/predictRN`

#### Entrada:

```json
{
  "horas": 10,
  "dificultad": 7,
  "estimated_time": 1,
  "dias_restantes": 5,
  "tasks": 3
}
```

#### Salida:

```json
{
  "prioridad": 1,
  "horas_recomendadas": 0.25
}
```

---

## 🧠 Descripción

El modelo de red neuronal predice:

* Prioridad de la tarea (0: baja, 1: media, 2: alta)
* Horas recomendadas de estudio

Estas predicciones son utilizadas por el backend principal (NestJS) para generar planes de estudio inteligentes mediante integración con modelos de lenguaje (LLM).

---

## 🧠 Entrenamiento del modelo

El archivo del modelo (`modelo_estudio.h5`) no está incluido en el repositorio.

Para generarlo, ejecuta:

```bash
python train_model.py
```

Esto creará el modelo entrenado necesario para que la API funcione correctamente.

---


## ⚠️ Notas

* El modelo (`modelo_estudio.h5`) debe estar en la raíz del proyecto.
* La API está diseñada para integrarse con un backend externo.

---
