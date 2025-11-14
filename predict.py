import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- 1. 基本設定 ---
MODEL_PATH = 'finetuned_best_model.keras'
IMG_SIZE = (224, 224) # train.py と全く同じサイズ
CLASS_NAMES = ['good', 'bad'] # train.py と全く同じクラス順

# ResNet50 の入力に必要な前処理関数をインポート
preprocess_input = tf.keras.applications.resnet50.preprocess_input

def load_and_preprocess_image(image_path):
    """
    画像を1枚読み込み、モデルの入力形式に前処理する
    """
    # 1. 画像の読み込みとリサイズ
    # target_size で自動的に (224, 224) にリサイズされる
    img = load_img(image_path, target_size=IMG_SIZE)
    
    # 2. Numpy配列に変換 (224, 224, 3)
    img_array = img_to_array(img)
    
    # 3. ResNet50用の前処理 (ピクセル値を-1〜1にスケーリング)
    img_preprocessed = preprocess_input(img_array)
    
    # 4. バッチ次元の追加 (1, 224, 224, 3)
    # model.predict は常にバッチでの入力を期待するため
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

def main():
    # --- 2. 実行時の引数チェック ---
    if len(sys.argv) != 2:
        print("使い方: python predict.py [判定したい画像ファイルのパス]")
        sys.exit(1)
        
    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        sys.exit(1)

    try:
        # --- 3. モデルの読み込み ---
        print(f"'{MODEL_PATH}' を読み込んでいます...")
        # compile=False にすると、予測専用として高速に読み込めます
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
        
        # --- 4. 画像の前処理 ---
        print(f"画像を前処理中: {image_path}")
        processed_image = load_and_preprocess_image(image_path)
        
        # --- 5. 予測の実行 ---
        print("予測を実行中...")
        prediction = model.predict(processed_image)
        
        # --- 6. 結果の解釈と表示 ---
        # 出力は (1, 1) の形状で、sigmoid (0〜1) の値が入っている
        # 0に近いほど 'good' (クラス0), 1に近いほど 'bad' (クラス1)
        score = prediction[0][0] 
        
        threshold = 0.8 # 判定の境界値
        
        if score < threshold:
            predicted_class = CLASS_NAMES[0] # 'good'
            confidence = 1 - score
        else:
            predicted_class = CLASS_NAMES[1] # 'bad'
            confidence = score
            
        print("\n" + "="*30)
        print("    予測結果")
        print("="*30)
        print(f"  ファイル: {os.path.basename(image_path)}")
        print(f"  判定:  ** {predicted_class.upper()} **")
        print(f"  信頼度: {confidence:.2%}")
        print(f"  (Rawスコア: {score:.4f}  *1に近いほど'bad')")
        print("="*30)

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        print("ヒント: 'Pillow' がインストールされていますか？ (pip install Pillow)")

if __name__ == "__main__":
    main()