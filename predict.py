import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- 1. 基本設定 ---
MODEL_PATH = 'v12_efficientnet_loss_best.keras'
IMG_SIZE = (224, 224) 
CLASS_NAMES = ['good', 'bad']

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch

def main():
    if len(sys.argv) != 2:
        print("使い方: python predict.py [判定したい画像ファイルのパス]")
        sys.exit(1)
        
    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        sys.exit(1)

    try:
        print(f"'{MODEL_PATH}' を読み込んでいます...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
        
        print(f"画像を前処理中: {image_path}")
        processed_image = load_and_preprocess_image(image_path)
        
        print("予測を実行中...")
        prediction = model.predict(processed_image)
        
        score = prediction[0][0] 
        
        # ★ 修正: 推奨閾値 0.4 に変更
        threshold = 0.4 
        
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
        print(f"  (Rawスコア: {score:.4f} / 閾値: {threshold})")
        print("="*30)

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()