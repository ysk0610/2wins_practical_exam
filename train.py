import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# --- 1. 定数と基本設定 ---
# ★ Mac用に相対パスに戻しました
TRAIN_DIR = 'dataset_split/train'
VAL_DIR = 'dataset_split/validation'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# --- 2. データ読み込み ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', class_names=['good', 'bad']
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', class_names=['good', 'bad'], shuffle=False
)

# --- 3. データ拡張 (V11仕様) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"), 
    tf.keras.layers.RandomRotation(0.2), 
    tf.keras.layers.RandomContrast(0.05),   
    tf.keras.layers.RandomBrightness(0.05), 
    tf.keras.layers.GaussianNoise(0.05),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
def augment_data(image, label):
    return data_augmentation(image, training=True), label

train_ds = train_ds.map(augment_data, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 4. クラス重み ---
# (good:800, bad:280 の場合)
class_weight = {0: 0.68, 1: 1.93}

# --- 5. モデル構築 (EfficientNetV2B0) ---
base_model = tf.keras.applications.EfficientNetV2B0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# --- 6. コンパイルとスケジューラ ---
LR_START, LR_MAX, LR_MIN = 0.00001, 0.001, 0.00001
WARMUP_EPOCHS = 3

def lr_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        lr = LR_START + (LR_MAX - LR_START) * (epoch / WARMUP_EPOCHS)
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    return lr

lr_callback = LearningRateScheduler(lr_schedule)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
)

# --- 7. 実行 ---
checkpoint = ModelCheckpoint(
    filepath='v11_efficientnet_scheduled_base.keras', # 保存名
    monitor='val_loss', mode='min', save_best_only=True
)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)

print("Starting Training...")
history = model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds, class_weight=class_weight,
    callbacks=[checkpoint, early_stopping, lr_callback]
)