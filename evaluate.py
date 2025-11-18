import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 設定 ---
MODEL_PATH = 'v12_efficientnet_loss_best.keras'
VAL_DIR = 'dataset_split/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['good', 'bad']

# ★ 最終決定した推奨閾値
THRESHOLD = 0.4 

# ★ EfficientNetV2用の前処理
preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

def main():
    # モデルファイルの存在確認
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルが見つかりません {MODEL_PATH}")
        print("Google Driveからモデルをダウンロードして配置してください。")
        return

    # 1. モデルの読み込み
    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 2. 検証データの読み込み
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='binary', class_names=CLASS_NAMES, shuffle=False
    )
    
    # 3. 正解ラベルと予測スコアを取得
    y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()
    
    print("Predicting...")
    y_pred_proba = model.predict(val_ds).flatten()
    
    # 4. 閾値判定 (0.40以上なら bad=1 とする)
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)

    # 5. 最終レポートの出力
    print("\n" + "="*60)
    print(f" 🏆 最終評価レポート (Threshold = {THRESHOLD}) ")
    print("="*60)
    
    # 分類レポート (Precision, Recall, F1-score)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # 混同行列の数値表示
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print(f"( {CLASS_NAMES[0]} / {CLASS_NAMES[1]} の順 )")

    # 6. 混同行列の可視化と保存
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (Threshold = {THRESHOLD})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_name = 'evaluation_matrix_final.png'
    plt.savefig(save_name)
    print("-" * 60)
    print(f"混同行列の画像を '{save_name}' に保存しました。")

if __name__ == "__main__":
    main()