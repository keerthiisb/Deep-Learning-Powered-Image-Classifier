import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # Change to EfficientNetV2 if needed
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageFile
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # ✅ Import tqdm for progress tracking

# ✅ Enable PIL to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ GPU Check
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"✅ Using GPU: {gpus[0]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU detected! Training on CPU.")

# ✅ Paths
dataset_path = r"C:\Users\vigne\OneDrive\Desktop\CNN\dataset"
excel_file = r"C:\Users\vigne\OneDrive\Desktop\CNN\input.xlsx"
output_csv = r"C:\Users\vigne\OneDrive\Desktop\CNN\results.csv"
model_dir = r"C:\Users\vigne\OneDrive\Desktop\CNN\models"
os.makedirs(model_dir, exist_ok=True)

# ✅ Image Settings
image_size = (224, 224)
batch_size = 16
epochs = 50  # Increased for better training

# ✅ Ask User for Model Selection
use_existing = input("Do you want to use a previously trained model? (yes/no): ").strip().lower()
model_path = None

if use_existing == 'yes':
    available_models = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if available_models:
        print("Available models:")
        for i, model_name in enumerate(available_models):
            print(f"{i + 1}. {model_name}")
        model_choice = int(input("Select a model (1, 2, ...): ")) - 1
        model_path = os.path.join(model_dir, available_models[model_choice])
    else:
        print("No trained models found. Training a new model.")
        use_existing = 'no'

if use_existing == 'no':
    model_name = input("Enter a name for the new model: ").strip() + ".h5"
    model_path = os.path.join(model_dir, model_name)

# ✅ Count Images per Class
class_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))}
total_images = sum(class_counts.values())

# ✅ Adaptive Training
validation_split = 0.2 if total_images >= 50 else 0.0
batch_size = 8 if total_images < 50 else batch_size

# ✅ Data Augmentation (Enhanced)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # NEW: Better perspective transformations
    zoom_range=0.3,   # NEW: Random zoom
    brightness_range=[0.8, 1.2],  # NEW: Brightness variations
    horizontal_flip=True,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
) if validation_split > 0 else None

num_classes = len(train_generator.class_indices)

# ✅ Load or Train Model
if use_existing == 'yes':
    try:
        with custom_object_scope({}):
            model = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded existing model: {model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}. Training a new model instead.")
        use_existing = 'no'

if use_existing == 'no':
    # ✅ Use EfficientNetV2 instead for higher accuracy (Optional)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # base_model = tf.keras.applications.EfficientNetV2B0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')  # (Uncomment to use)

    # ✅ Unfreeze last 20 layers for fine-tuning
    for layer in base_model.layers[:-20]:  
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),  # NEW: Helps reduce overfitting
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.6),  # NEW: Stronger dropout for regularization
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),  # NEW: Learning rate optimized
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ✅ Class Weights for Balanced Training
    class_weights = {i: total_images / (num_classes * count) if count > 0 else 1.0 for i, count in enumerate(class_counts.values())}

    # ✅ Callbacks for Accuracy
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    print("✅ Training model...")
    if validation_generator:
        model.fit(train_generator, epochs=epochs, validation_data=validation_generator, class_weight=class_weights, callbacks=[early_stopping, reduce_lr])
    else:
        model.fit(train_generator, epochs=epochs, class_weight=class_weights, callbacks=[early_stopping, reduce_lr])

    model.save(model_path)
    print(f"✅ Model saved at {model_path}")

# ✅ Classify Images Efficiently
df_input = pd.read_excel(excel_file)

def classify_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_names = list(train_generator.class_indices.keys())
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        return predicted_class, confidence, "Success"
    except Exception as e:
        return "Error", 0.0, str(e)

# ✅ Fast Parallel Classification with Progress Tracking
results = []
total_images = len(df_input)
counter = 0

with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
        future_to_url = {executor.submit(classify_image, row['image link']): row for row in df_input.to_dict(orient='records')}
        for future in future_to_url:
            result = future.result()
            results.append(result)
            counter += 1
            pbar.update(1)  # Update progress bar
            print(f"✅ Processed: {counter}/{total_images}", end="\r")  # Inline update

df_input['Predicted_Class'], df_input['Confidence'], df_input['Status'] = zip(*results)
df_input.to_csv(output_csv, index=False)

print(f"\n✅ Final results saved to {output_csv}")
