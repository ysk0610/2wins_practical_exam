import tensorflow as tf
import numpy as np
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 設定 ---
MODEL_PATH = 'balanced_best_model.keras' # ★ 評価したいモデル
VAL_DIR = 'dataset_split/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 # 学習時と同じバッチサイズ
CLASS_NAMES = ['good', 'bad']

# ★ 評価したい閾値（しきい値）
# 0.5 が厳しそうなら 0.4 や 0.3 に変更して再実行する
THRESHOLD = 0.4

# ResNet50 の前処理関数
preprocess_input = tf.keras.applications.resnet50.preprocess_input

def preprocess_image(image, label):
    """
    Kerasデータセット用の前処理関数 (train.pyとは異なり、正規化のみ)
    """
    image = preprocess_input(image)
    return image, label

def main():
    try:
        # --- 2. モデルの読み込み ---
        print(f"'{MODEL_PATH}' を読み込んでいます...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # --- 3. 検証データセットの読み込み ---
        # 重要: shuffle=False にして、ラベルの順序が崩れないようにする
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary',
            class_names=CLASS_NAMES,
            shuffle=False 
        )
        
        # 3.1 正解ラベルを取得 (y_true)
        # val_ds からラベルだけを抽出し、Numpy配列に変換
        y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()
        print(f"検証データ {len(y_true)} 件を読み込みました。")

        # --- 4. 全データに対して予測を実行 ---
        print("検証データ全体に対して予測を実行中...")
        # 元の val_ds (前処理なし) をそのまま渡す
        # (モデル内部で前処理が自動的に行われるため)
        y_pred_proba = model.predict(val_ds).flatten() # 予測確率 (0.0〜1.0)(val_ds_processed).flatten() # 予測確率 (0.0〜1.0)
        
        # --- 5. 閾値(THRESHOLD)を適用して 0 or 1 に変換 ---
        y_pred = (y_pred_proba >= THRESHOLD).astype(int)

        # --- 6. 最終レポートの出力 ---
        print("\n" + "="*50)
        print(f"    最終評価レポート (閾値 = {THRESHOLD})")
        print("="*50)
        
        # 6.1 分類レポート (Precision / Recall / F1-score)
        # target_names を指定すると 'good' 'bad' で表示される
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        print("\n■ 分類レポート:")
        print(report)

        # 6.2 混同行列 (Confusion Matrix)
        cm = confusion_matrix(y_true, y_pred)
        print("\n■ 混同行列:")
        print(cm)
        print(f" ( {CLASS_NAMES[0]} / {CLASS_NAMES[1]} の順 )")
        
        # --- 7. 混同行列の可視化 ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix (Threshold = {THRESHOLD})')
        plt.ylabel('True Label (正解)')
        plt.xlabel('Predicted Label (予測)')
        
        # グラフを画像ファイルとして保存
        plt.savefig('evaluation_confusion_matrix.png')
        print(f"\n混同行列グラフを 'evaluation_confusion_matrix.png' として保存しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("ヒント: 'scikit-learn' や 'seaborn' はインストールされていますか？")
        print("( pip install scikit-learn seaborn )")

if __name__ == "__main__":
    main()