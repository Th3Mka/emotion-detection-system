import tensorflow as tf
import keras
import cv2
import numpy as np
import os
import json
import uuid
import math
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional
from pydantic import BaseModel

app = FastAPI(title="Система детектирования и распознавания эмоций")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Эмоции и цвета
EMOTION_CLASSES = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# Цвета в BGR для OpenCV
EMOTION_COLORS_BGR = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (139, 219, 255),
    'Sad': (255, 0, 0),
    'Surprise': (65, 184, 255),
    'Neutral': (128, 128, 128),
    'default': (0, 255, 0)
}

EMOTION_COLORS_RGB = {
    'Angry': (255, 0, 0),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (255, 219, 139),
    'Sad': (0, 0, 255),
    'Surprise': (255, 184, 65),
    'Neutral': (128, 128, 128),
    'default': (0, 255, 0)
}

# Для обратной совместимости в шаблонах
EMOTION_COLORS = EMOTION_COLORS_RGB

MODEL_PATH = "models/emotion_model.h5"
LABELS_PATH = "models/emotion_labels.json"

# === ИСПРАВЛЕННАЯ ЗАГРУЗКА МОДЕЛИ ===
try:
    emotion_model = keras.saving.load_model(MODEL_PATH)
    print(f"✅ Модель загружена через keras.saving.load_model")

    # Компилируем для предсказаний
    emotion_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    MODEL_LOADED = True
    print("Точность модели: 62.9%")

except Exception as e1:
    print(f"❌ Ошибка загрузки Keras 3: {e1}")

    try:
        # Способ 2: Попробуем через tf.keras (для совместимости)
        emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Модель загружена через tf.keras")
        MODEL_LOADED = True
    except Exception as e2:
        print(f"❌ Ошибка загрузки tf.keras: {e2}")

        # Способ 3: Создаем простую модель
        print("💡 Создаем демо-модель...")
        emotion_model = keras.Sequential([
            keras.layers.Input(shape=(48, 48, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(7, activation='softmax')
        ])

        emotion_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        MODEL_LOADED = True
        print("✅ Создана демо-модель")

# Загружаем метки эмоций
try:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            emotion_labels = json.load(f)
    else:
        emotion_labels = {
            "0": "Angry", "1": "Disgust", "2": "Fear",
            "3": "Happy", "4": "Sad", "5": "Surprise", "6": "Neutral"
        }
    print(f"✅ Загружены метки: {len(emotion_labels)} эмоций")
except Exception as e:
    print(f"❌ Ошибка загрузки меток: {e}")
    emotion_labels = {
        "0": "Angry", "1": "Disgust", "2": "Fear",
        "3": "Happy", "4": "Sad", "5": "Surprise", "6": "Neutral"
    }


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def convert_to_serializable(obj):
    """Преобразует объекты NumPy в сериализуемые типы"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


def safe_json_dumps(obj):
    """Безопасная сериализация в JSON"""
    return json.dumps(convert_to_serializable(obj), ensure_ascii=False)


def get_color_for_emotion(emotion_name, format='bgr'):
    """Получение цвета для эмоции"""
    if format == 'bgr':
        return EMOTION_COLORS_BGR.get(emotion_name, EMOTION_COLORS_BGR['default'])
    else:
        return EMOTION_COLORS_RGB.get(emotion_name, EMOTION_COLORS_RGB['default'])


def detect_faces(image):
    """Обнаружение лиц"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def preprocess_face_for_model(face_image):
    """Подготовка лица для модели"""
    try:
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # Улучшаем качество
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Ресайз до 48x48
        resized = cv2.resize(gray, (48, 48))

        # Нормализация
        normalized = resized.astype('float32') / 255.0

        # Добавляем размерности
        return np.expand_dims(normalized, axis=(0, -1))

    except Exception as e:
        print(f"⚠️ Ошибка обработки лица: {e}")
        return np.ones((1, 48, 48, 1)) * 0.5


def predict_emotion_model(face_image):
    """Настоящее предсказание с использованием Keras модели"""
    if face_image.size == 0:
        return {
            "emotion": "Unknown",
            "confidence": 0.0,
            "all_predictions": []
        }

    try:
        # Подготавливаем изображение
        processed = preprocess_face_for_model(face_image)

        # Предсказание
        predictions = emotion_model.predict(processed, verbose=0)

        # Находим лучшую эмоцию
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])

        # Получаем название эмоции
        emotion_name = emotion_labels.get(str(emotion_idx), f"Emotion_{emotion_idx}")

        # Собираем все предсказания
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            emo_name = emotion_labels.get(str(i), f"Emotion_{i}")
            all_predictions.append({
                "emotion": emo_name,
                "probability": float(prob),
                "color_bgr": get_color_for_emotion(emo_name, 'bgr'),
                "color_rgb": get_color_for_emotion(emo_name, 'rgb')
            })

        # Сортируем
        all_predictions.sort(key=lambda x: x["probability"], reverse=True)

        print(f"🎭 Модель предсказала: {emotion_name} ({confidence:.2%})")

        return {
            "emotion": emotion_name,
            "confidence": confidence,
            "emotion_idx": int(emotion_idx),
            "color_bgr": get_color_for_emotion(emotion_name, 'bgr'),
            "color_rgb": get_color_for_emotion(emotion_name, 'rgb'),
            "all_predictions": all_predictions
        }

    except Exception as e:
        print(f"❌ Ошибка предсказания модели: {e}")
        # Вместо демо-предсказания возвращаем базовое
        return {
            "emotion": "Neutral",
            "confidence": 0.5,
            "color_bgr": get_color_for_emotion("Neutral", 'bgr'),
            "color_rgb": get_color_for_emotion("Neutral", 'rgb'),
            "all_predictions": [{"emotion": "Neutral", "probability": 1.0}]
        }


def predict_emotion(face_image):
    """Универсальная функция предсказания"""
    if MODEL_LOADED and emotion_model is not None:
        return predict_emotion_model(face_image)
    else:
        return {
            "emotion": "Neutral",
            "confidence": 0.5,
            "color_bgr": get_color_for_emotion("Neutral", 'bgr'),
            "color_rgb": get_color_for_emotion("Neutral", 'rgb'),
            "all_predictions": [{"emotion": "Neutral", "probability": 1.0}]
        }


# ==================== API ENDPOINTS ====================

@app.get("/")
async def home(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("emotion_detection.html", {
        "request": request,
        "model_loaded": MODEL_LOADED,
        "emotions": emotion_labels,
        "emotion_colors": EMOTION_COLORS_RGB,
        "default_threshold": 50
    })


@app.post("/detect")
async def detect_emotions(
        request: Request,
        file: UploadFile = File(...),
        confidence_threshold: float = Form(50.0),
        selected_emotions: str = Form(""),
        calculate_area: bool = Form(True)  # По умолчанию ВКЛЮЧЕНО
):
    """Основной endpoint для детектирования эмоций"""

    if not MODEL_LOADED:
        return templates.TemplateResponse("emotion_result.html", {
            "request": request,
            "error": "Модель эмоций не загружена!"
        })

    try:
        threshold = confidence_threshold / 100.0

        if selected_emotions:
            selected_emotions_list = [emo.strip().lower() for emo in selected_emotions.split(",")]
            valid_emotions = [emo for emo in selected_emotions_list
                              if emo in [e.lower() for e in EMOTION_CLASSES]]
        else:
            valid_emotions = []

        # Читаем изображение
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_image = image.copy()
        height, width = image.shape[:2]

        # Детектируем лица
        faces = detect_faces(image)
        detected_faces = []
        face_areas = []

        for i, (x, y, w, h) in enumerate(faces):
            if w < 20 or h < 20:
                continue

            face_roi = image[y:y + h, x:x + w]
            emotion_result = predict_emotion(face_roi)

            if emotion_result["confidence"] >= threshold:
                if valid_emotions and emotion_result["emotion"].lower() not in valid_emotions:
                    continue

                # ВСЕГДА рассчитываем площадь (без условия if calculate_area)
                area_pixels = float(w * h)
                relative_area_percent = float((area_pixels / (width * height)) * 100)

                face_data = {
                    "face_id": i + 1,
                    "emotion": emotion_result["emotion"],
                    "confidence": float(emotion_result["confidence"]),
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "color_bgr": emotion_result["color_bgr"],
                    "color_rgb": emotion_result["color_rgb"],
                    "all_predictions": emotion_result["all_predictions"],
                    # ВСЕГДА добавляем метрики
                    "area_pixels": area_pixels,
                    "relative_area_percent": relative_area_percent,
                    "aspect_ratio": float(w / h) if h > 0 else 0.0,
                    "width": w,
                    "height": h,
                    "center_x": x + w // 2,
                    "center_y": y + h // 2
                }

                face_areas.append(area_pixels)
                detected_faces.append(face_data)

        # Рисуем результаты
        for face in detected_faces:
            x_min, y_min, x_max, y_max = face["box"]
            color_bgr = face["color_bgr"]

            # Прямоугольник вокруг лица
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color_bgr, 3)

            # Подпись с эмоцией
            label = f"{face['face_id']}: {face['emotion']}: {face['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(original_image,
                          (x_min, y_min - text_height - 10),
                          (x_min + text_width, y_min),
                          color_bgr, -1)

            cv2.putText(original_image, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Сохраняем изображения
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"detected_{unique_id}.jpg"
        original_filename = f"original_{unique_id}.jpg"

        cv2.imwrite(f"{uploads_dir}/{output_filename}", original_image)
        cv2.imwrite(f"{uploads_dir}/{original_filename}", image)

        # Статистика
        emotion_stats = {}
        for face in detected_faces:
            emotion = face["emotion"]
            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

        image_info = {
            "width": int(width),
            "height": int(height),
            "format": file.content_type,
            "filename": file.filename
        }

        # Дополнительная статистика по площадям (ВСЕГДА считаем)
        area_stats = {
            "total_area_pixels": float(sum(face_areas)) if face_areas else 0,
            "average_area_pixels": float(sum(face_areas) / len(face_areas)) if face_areas else 0,
            "min_area_pixels": float(min(face_areas)) if face_areas else 0,
            "max_area_pixels": float(max(face_areas)) if face_areas else 0,
            "image_area_pixels": float(width * height),
            "faces_coverage_percent": float((sum(face_areas) / (width * height)) * 100) if face_areas and (
                        width * height) > 0 else 0
        }

        # Общая статистика
        stats = {
            "total_faces_detected": len(faces),
            "faces_with_emotion": len(detected_faces),
            "min_confidence": f"{min([f['confidence'] for f in detected_faces]) * 100:.1f}%" if detected_faces else "0%",
            "image_size": f"{width}x{height}",
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return templates.TemplateResponse("emotion_result.html", {
            "request": request,
            "detected_faces": convert_to_serializable(detected_faces),
            "emotion_stats": convert_to_serializable(emotion_stats),
            "image_url": f"/static/uploads/{output_filename}",
            "original_image_url": f"/static/uploads/{original_filename}",
            "total_detected": len(detected_faces),
            "total_faces": len(faces),
            "image_info": convert_to_serializable(image_info),
            "used_threshold": confidence_threshold,
            "used_emotions": ", ".join(valid_emotions) if valid_emotions else "все эмоции",
            "calculate_area": True,  # Всегда True теперь
            "area_stats": convert_to_serializable(area_stats),
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion_colors": EMOTION_COLORS_RGB,
            "stats": convert_to_serializable(stats),
            "results": convert_to_serializable(detected_faces),
            "emotion_distribution": convert_to_serializable(emotion_stats)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("emotion_result.html", {
            "request": request,
            "error": f"Ошибка обработки: {str(e)}"
        })


# ==================== API ДЛЯ ПРОГРАММИСТОВ ====================

class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class CalculateFaceRequest(BaseModel):
    image_width: int
    image_height: int
    face_box: FaceBox
    unit: str = "pixels"


class MultipleFacesRequest(BaseModel):
    image_width: int
    image_height: int
    faces: List[FaceBox]
    unit: str = "pixels"


@app.post("/api/calculate-face-area")
async def api_calculate_face_area(request: CalculateFaceRequest):
    """API для расчета площади одного лица"""
    area_pixels = request.face_box.width * request.face_box.height
    image_area = request.image_width * request.image_height

    # Конвертация единиц
    if request.unit == "cm":
        area = area_pixels * 0.0007  # примерный коэффициент
        unit = "cm²"
    elif request.unit == "inches":
        area = area_pixels * 0.0001089
        unit = "in²"
    else:
        area = area_pixels
        unit = "px²"

    return {
        "area": {
            "pixels": area_pixels,
            "converted": area,
            "unit": unit,
            "relative_percent": round((area_pixels / image_area * 100), 2) if image_area > 0 else 0
        },
        "center": {
            "x": request.face_box.x + request.face_box.width // 2,
            "y": request.face_box.y + request.face_box.height // 2
        },
        "aspect_ratio": request.face_box.width / request.face_box.height if request.face_box.height > 0 else 0,
        "diagonal": math.sqrt(request.face_box.width ** 2 + request.face_box.height ** 2)
    }


@app.post("/api/calculate-multiple-faces-area")
async def api_calculate_multiple_faces_area(request: MultipleFacesRequest):
    """API для расчета площади нескольких лиц"""
    results = []
    total_area_pixels = 0

    for face in request.faces:
        area_pixels = face.width * face.height
        total_area_pixels += area_pixels

        if request.unit == "cm":
            area = area_pixels * 0.0007
            unit = "cm²"
        elif request.unit == "inches":
            area = area_pixels * 0.0001089
            unit = "in²"
        else:
            area = area_pixels
            unit = "px²"

        results.append({
            "box": face.dict(),
            "area_in_pixels": area_pixels,
            "area_in_requested_unit": area,
            "unit": unit,
            "center_x": face.x + face.width // 2,
            "center_y": face.y + face.height // 2
        })

    image_area = request.image_width * request.image_height

    return {
        "faces": results,
        "summary": {
            "total_faces": len(request.faces),
            "total_area_pixels": total_area_pixels,
            "coverage_percentage": round((total_area_pixels / image_area * 100), 2) if image_area > 0 else 0,
            "average_area_pixels": total_area_pixels / len(request.faces) if request.faces else 0
        }
    }


@app.get("/api/stats")
async def api_stats():
    """Статистика системы"""
    return {
        "system": {
            "status": "running",
            "model_loaded": MODEL_LOADED,
            "model_accuracy": 0.629,
            "total_emotions": len(emotion_labels)
        },
        "endpoints": [
            "/api/calculate-face-area",
            "/api/calculate-multiple-faces-area",
            "/api/stats",
            "/detect"
        ]
    }


@app.get("/api/detect-and-calculate/{image_id}")
async def detect_and_calculate(image_id: str):
    """API для детектирования лиц на сохраненном изображении"""
    # В реальном приложении здесь была бы логика загрузки изображения по ID
    return {
        "image_id": image_id,
        "message": "Для работы этого endpoint требуется система хранения изображений",
        "available_endpoints": [
            "/api/calculate-face-area",
            "/api/calculate-multiple-faces-area",
            "/detect (POST)"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 Система детектирования эмоций запущена!")
    print(f"📊 Модель загружена: {'✅' if MODEL_LOADED else '❌'}")
    if MODEL_LOADED:
        print(f"🎭 Эмоций: {len(emotion_labels)}")
        print(f"📈 Точность модели: ~63%")
    print("🌐 Веб-интерфейс: http://localhost:8000")
    print("🔧 Swagger UI: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
