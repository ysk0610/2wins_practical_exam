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
EPOCHS = 10 

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


# --- 5. モデル構築（転移学習 ResNet50） ---
# ResNet50の「画像から特徴を抽出する部分」だけを読み込む
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',      # ImageNetで事前学習済みの重みを使う
    include_top=False,       # ImageNet用の分類層（1000クラス）は不要
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# ベースモデルの重みを「凍結」し、学習で更新されないようにする
base_model.trainable = False

# ResNet50の入力に必要な前処理（ピクセル値を-1〜1にスケーリング）
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# モデルの全体を定義
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = preprocess_input(inputs) # 1. 前処理
x = base_model(x, training=False) # 2. ベースモデル（凍結）
x = tf.keras.layers.GlobalAveragePooling2D()(x) # 3. 特徴を平坦化
x = tf.keras.layers.Dense(128, activation='relu')(x) # 4. 独自の学習層
x = tf.keras.layers.Dropout(0.5)(x) # 5. 過学習防止
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # 6. 出力層 (0〜1の値)

model = tf.keras.Model(inputs, outputs)

model.summary() # モデルの構造を表示


# --- 6. モデルのコンパイル ---
# [cite_start]最重要KPIである Recall（再現率） を監視対象に設定 [cite: 26]
metrics = [
    'accuracy',
    tf.keras.metrics.Recall(name='recall'), # 不良品 ('bad'=1) の見逃し率
    tf.keras.metrics.Precision(name='precision') # 不良品判定の精度
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', # 二値分類の標準的な損失関数
    metrics=metrics
)

#ベストモデルを保存する
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # 保存するファイル名
    monitor='val_accuracy',       # 監視する指標
    mode='max',                   # 'val_accuracy' なので最大(max)を目指す
    save_best_only=True           # ベストなものだけを保存する
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