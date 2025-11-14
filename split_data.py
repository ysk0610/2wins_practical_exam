import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(base_dir="dataset", output_dir="dataset_split", test_size=0.2):
    """
    元のデータセットを層化サンプリングして、train/validation フォルダに分割・コピーする。
    """
    
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # 元画像のパスとラベル（クラス名）をすべて取得
    filepaths = []
    labels = []
    
    for class_dir in base_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for file in class_dir.glob("*.png"): # .png 以外も対象なら "*.jpg" なども追加
                filepaths.append(file)
                labels.append(class_name)
    
    if not filepaths:
        print(f"エラー: '{base_dir}' フォルダに画像が見つかりません。")
        print("（dataset フォルダが 2wins_practical_exam の直下にあるか確認してください）")
        return

    print(f"合計画像数: {len(filepaths)} 枚 (内訳: { {label: labels.count(label) for label in set(labels)} })")

    # 1. 層化サンプリングで分割
    # stratify=labels にすることで、labels（good/bad）の比率を保ったまま分割する
    train_files, val_files, train_labels, val_labels = train_test_split(
        filepaths, 
        labels, 
        test_size=test_size, 
        random_state=42,  # 再現性のために乱数シードを固定
        stratify=labels
    )
    
    print(f"学習データ数: {len(train_files)} 枚")
    print(f"検証データ数: {len(val_files)} 枚")

    # 2. 出力用ディレクトリを作成
    output_path.mkdir(exist_ok=True)
    
    train_path = output_path / "train"
    val_path = output_path / "validation"
    
    # 既存の出力フォルダがあれば一度削除（安全のため）
    if train_path.exists():
        shutil.rmtree(train_path)
    if val_path.exists():
        shutil.rmtree(val_path)
        
    train_path.mkdir()
    val_path.mkdir()
    
    for label in set(labels):
        (train_path / label).mkdir()
        (val_path / label).mkdir()

    # 3. ファイルを新しい場所にコピー
    def copy_files(files, labels, dest_dir):
        for file, label in zip(files, labels):
            dest_file = dest_dir / label / file.name
            shutil.copy(file, dest_file)

    copy_files(train_files, train_labels, train_path)
    copy_files(val_files, val_labels, val_path)
    
    print("-" * 30)
    print(f"分割完了！ '{output_dir}' フォルダを確認してください。")
    print(f"学習用 (train): { {label: len(list((train_path / label).glob('*.png'))) for label in set(labels)} }")
    print(f"検証用 (val):   { {label: len(list((val_path / label).glob('*.png'))) for label in set(labels)} }")


if __name__ == "__main__":
    # dataset フォルダがこのスクリプトと同じ階層にあることを想定
    split_dataset(base_dir="dataset", test_size=0.2)