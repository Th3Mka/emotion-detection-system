import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split

# Параметры
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 7

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


def load_fer2013_dataset(csv_path='fer2013/fer2013.csv'):
    """Загрузка датасета FER2013"""
    df = pd.read_csv(csv_path)

    # Разделение на пиксели и метки
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()
    usage = df['Usage'].tolist()

    # Преобразование пикселей в изображения
    images = []
    for pixel_sequence in pixels:
        pixels_array = np.array(pixel_sequence.split(' '), dtype=np.float32)
        image = pixels_array.reshape(IMG_SIZE, IMG_SIZE)
        images.append(image)

    images = np.array(images) / 255.0  # Нормализация
    images = np.expand_dims(images, -1)  # Добавляем канал
    emotions = np.array(emotions)

    # Разделение на train/val/test
    train_indices = [i for i, u in enumerate(usage) if u == 'Training']
    val_indices = [i for i, u in enumerate(usage) if u == 'PublicTest']
    test_indices = [i for i, u in enumerate(usage) if u == 'PrivateTest']

    X_train = images[train_indices]
    y_train = emotions[train_indices]
    X_val = images[val_indices]
    y_val = emotions[val_indices]
    X_test = images[test_indices]
    y_test = emotions[test_indices]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_emotion_model():
    """Создание модели для распознавания эмоций"""

    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Второй сверточный блок
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Третий сверточный блок
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def create_data_augmentation():
    """Создание генератора аугментации данных"""
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def train():
    """Основная функция обучения"""

    print("📊 Загрузка датасета FER2013...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013_dataset()

    print(f"📈 Данные загружены:")
    print(f"   Обучающие: {X_train.shape}")
    print(f"   Валидационные: {X_val.shape}")
    print(f"   Тестовые: {X_test.shape}")

    # Создание модели
    print("🤖 Создание модели...")
    model = create_emotion_model()

    # Компиляция
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Коллбэки
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        keras.callbacks.ModelCheckpoint(
            'models/best_emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Аугментация данных
    datagen = create_data_augmentation()
    train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

    # Обучение
    print("🎯 Начало обучения...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Сохранение модели
    model.save('models/emotion_model.h5')
    print("💾 Модель сохранена как 'models/emotion_model.h5'")

    # Сохранение меток классов
    with open('models/emotion_labels.json', 'w') as f:
        json.dump(EMOTIONS, f)
    print("💾 Метки классов сохранены в 'models/emotion_labels.json'")

    # Оценка на тестовых данных
    print("\n📊 Оценка на тестовых данных:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Точность: {test_acc:.4f}")
    print(f"   Потери: {test_loss:.4f}")

    # Визуализация результатов
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # График точности
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Точность модели')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Точность')
    axes[0].legend()
    axes[0].grid(True)

    # График потерь
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Потери модели')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Потери')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Создание директорий
    os.makedirs('models', exist_ok=True)
    os.makedirs('fer2013', exist_ok=True)

    print("🚀 Начало обучения модели распознавания эмоций")
    print("=" * 50)

    try:
        model, history = train()
        print("\n✅ Обучение завершено успешно!")

        # Дополнительная информация
        model.summary()

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        print("\n💡 Замечание: Для обучения скачайте датасет FER2013")
        print("   и разместите файл fer2013.csv в папке fer2013/")