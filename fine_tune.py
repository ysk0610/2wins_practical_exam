import tensorflow as tf
import matplotlib.pyplot as plt
import os
# tensorflow.keras から色々 import しているところに、以下を追加
from tensorflow.keras.callbacks import ModelCheckpoint
# --- 1. 定数と基本設定 ---
TRAIN_DIR = 'dataset_split/train'
VAL_DIR = 'dataset_split/validation'
IMG_SIZE = (224, 224) # ResNet50の標準的な入力サイズにリサイズ
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = 'best_model.keras'

# --- 2. データ読み込みと準備 ---
# image_dataset_from_directory を使うと、フォルダから自動でデータを読み込める
# class_names=['good', 'bad'] と明示的に指定することが重要！
# これにより、Kerasは 'good' をクラス 0, 'bad' をクラス 1 として扱います。
# したがって、KPIの「Recall」は 'bad' (クラス 1) の再現率を正しく計算します。

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary', # good/bad の二値分類
    class_names=['good', 'bad'] 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    class_names=['good', 'bad'],
    shuffle=False # 検証用データはシャッフルしない
)

print(f"クラス名: {train_ds.class_names} ('good'=0, 'bad'=1)")

# --- 3. データ拡張（Data Augmentation） ---
# Kerasのレイヤーとしてデータ拡張を定義
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# 学習用データにだけデータ拡張を適用する
# .map() を使い、データセットの処理パイプラインを構築
# AUTOTUNEで読み込みを高速化
AUTOTUNE = tf.data.AUTOTUNE

def augment_data(image, label):
    # データ拡張レイヤーを適用
    return data_augmentation(image, training=True), label

train_ds = train_ds.map(augment_data, num_parallel_calls=AUTOTUNE)

# データセットをメモリにプリフェッチしてパフォーマンスを最適化
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# --- 4. クラス重み付け（Class Weighting）の計算 ---
# [cite_start]不均衡データ対策 [cite: 9]
# 学習用データの内訳: 'good': 800, 'bad': 280
COUNT_GOOD = 800
COUNT_BAD = 280
total_count = COUNT_GOOD + COUNT_BAD

# 'bad' (クラス 1) の予測を間違えたときのペナルティを重くする
weight_for_0_good = (1 / COUNT_GOOD) * (total_count / 2.0)
weight_for_1_bad = (1 / COUNT_BAD) * (total_count / 2.0)

class_weight = {0: weight_for_0_good, 1: weight_for_1_bad}

print(f"クラス重み: 'good'(0)={weight_for_0_good:.2f}, 'bad'(1)={weight_for_1_bad:.2f}")


# --- 5. 学習済みモデルの読み込み ---
print(f"学習済みモデル '{MODEL_PATH}' を読み込んでいます...")
model = tf.keras.models.load_model(MODEL_PATH)

print("ロードしたモデルの構造:")
model.summary() # <-- 確認のために summary を表示

base_model = model.get_layer('resnet50')

# --- ファインチューニングのための設定 ---
# 1. ベースモデルの凍結を解除
base_model.trainable = True

# 2. ResNet50の深い層（例: 最後の10層）だけを学習対象にする
#    (バッチ正規化層は凍結したままにするのがコツです)
print(f"ResNet50の層の数: {len(base_model.layers)}")
fine_tune_at_layer = 165 # ResNet50の最後のブロック（conv5_block3）あたり

# 165層目より手前はすべて凍結
for layer in base_model.layers[:fine_tune_at_layer]:
    layer.trainable = False

print("ファインチューニング設定後のモデル構造:")
model.summary()


# --- 6. モデルのコンパイル ---
# [cite_start]最重要KPIである Recall（再現率） を監視対象に設定 [cite: 26]
metrics = [
    'accuracy',
    tf.keras.metrics.Recall(name='recall'), # 不良品 ('bad'=1) の見逃し率
    tf.keras.metrics.Precision(name='precision') # 不良品判定の精度
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # 学習率を 1/100 に変更！
    loss='binary_crossentropy',
    metrics=metrics
)

checkpoint = ModelCheckpoint(
    filepath='finetuned_best_model.keras', # 保存ファイル名を変更
    monitor='val_recall',                 # 監視対象をKPI（再現率）に変更
    mode='max',                           # 'val_recall' なので max を目指す
    save_best_only=True
)

# --- 7. 学習の実行 ---
print("\n" + "="*30)
print("学習を開始します...")
print("="*30 + "\n")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight, # ここでクラス重みを適用！
    callbacks=[checkpoint]
)

print("\n学習が完了しました。")


# --- 8. 結果の保存と可視化 ---

# 1. モデルの保存

# 2. 学習曲線のプロットと保存
# [cite_start]レポート提出用に結果を可視化 [cite: 15, 23]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
recall = history.history['recall']
val_recall = history.history['val_recall']

epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 8))

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
plt.title('Training and Validation Loss')

# 再現率 (Recall) - 最重要KPI
plt.subplot(2, 2, 3)
plt.plot(epochs_range, recall, label='Training Recall')
plt.plot(epochs_range, val_recall, label='Validation Recall')
plt.legend(loc='lower right')
plt.title('Training and Validation Recall (KPI)')
plt.ylim([0, 1.05]) # 0%から100%の範囲で表示

# 適合率 (Precision)
plt.subplot(2, 2, 4)
plt.plot(epochs_range, history.history['precision'], label='Training Precision')
plt.plot(epochs_range, history.history['val_precision'], label='Validation Precision')
plt.legend(loc='lower right')
plt.title('Training and Validation Precision')
plt.ylim([0, 1.05]) # 0%から100%の範囲で表示

plt.tight_layout()
# グラフを画像ファイルとして保存
plt.savefig('training_history.png')
print(f"学習グラフを 'training_history.png' として保存しました。")