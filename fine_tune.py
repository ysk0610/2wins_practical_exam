import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# --- 1. 設定 ---
TRAIN_DIR = 'dataset_split/train'
VAL_DIR = 'dataset_split/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 

# ★ 土台モデル名
MODEL_PATH = 'v11_efficientnet_scheduled_base.keras'

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
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
def augment_data(image, label):
    return data_augmentation(image, training=True), label

train_ds = train_ds.map(augment_data, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
class_weight = {0: 0.68, 1: 1.93}

# --- 5. モデル読み込み ---
if not os.path.exists(MODEL_PATH):
    print(f"エラー: 土台モデル {MODEL_PATH} が見つかりません。train.pyを実行してください。")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# EfficientNetV2B0 の層を取得して解凍
base_model = model.get_layer('efficientnetv2-b0')
base_model.trainable = True
fine_tune_at = 170
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# --- 6. スケジュール (Fine-Tuning用) ---
LR_START, LR_MAX, LR_MIN = 0.000001, 0.0001, 0.000001
WARMUP_EPOCHS = 3

def lr_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        lr = LR_START + (LR_MAX - LR_START) * (epoch / WARMUP_EPOCHS)
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    return lr
lr_callback = LearningRateScheduler(lr_schedule)

# --- 7. コンパイル ---
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-4), 
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
)

# --- 8. 実行 ---
checkpoint = ModelCheckpoint(
    filepath='v12_efficientnet_loss_best.keras', # ★最終モデル名
    monitor='val_loss', mode='min', save_best_only=True
)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

print("Starting Fine-Tuning...")
history = model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds, class_weight=class_weight,
    callbacks=[checkpoint, early_stopping, lr_callback]
)

# --- 9. 結果の保存と可視化 (V12 Fine-Tuning の証拠) ---

# EarlyStoppingで学習が途中で止まることを考慮し、実行されたエポック数でグラフを描画
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
recall = history.history['recall']
val_recall = history.history['val_recall']
precision = history.history['precision']
val_precision = history.history['val_precision']


epochs_range = range(len(loss)) # 実際に実行されたエポック数

plt.figure(figsize=(14, 10))

# 精度 (Accuracy)
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# 損失 (Loss)
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (V12 Base)')

# 再現率 (Recall) - 最重要KPI
plt.subplot(2, 2, 3)
plt.plot(epochs_range, recall, label='Training Recall')
plt.plot(epochs_range, val_recall, label='Validation Recall')
plt.legend(loc='lower right')
plt.title('Training and Validation Recall (KPI)')
plt.ylim([0, 1.05])

# 適合率 (Precision)
plt.subplot(2, 2, 4)
plt.plot(epochs_range, precision, label='Training Precision')
plt.plot(epochs_range, val_precision, label='Validation Precision')
plt.legend(loc='lower right')
plt.title('Training and Validation Precision')
plt.ylim([0, 1.05])

plt.tight_layout()
# ★ グラフを v12 の名前で保存
plt.savefig('training_history_v12.png')
print(f"学習グラフを 'training_history_v12.png' として保存しました。")