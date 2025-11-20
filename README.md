# 不良品識別モデル 最終報告 (EfficientNetV2採用)

このREADMEは、実技試験課題「不良品識別モデル」の最終報告レポートの要点を兼ねています。

## 1. プロジェクト概要と最終結論

| 項目 | 詳細 | 補足 (要件理解) |
| :--- | :--- | :--- |
| **課題** | 製造ラインにおける製品の外観検査自動化。 | 検査員の見落としリスクと熟練者依存の解消。 |
| **KPI** | **Recall (再現率) の最大化** | 不良品の見落としが後のトラブルに繋がるため、検知を最優先。 |
| **採用モデル** | **EfficientNetV2B0** | ResNet50の性能限界を突破するために採用した最新鋭のアーキテクチャ。 |
| **最終成果** | Recall **87.1%** / Precision **92.4%** | V3.4 (Recall 83% / Precision 54%) を大きく上回るバランスを達成。 |

***

## 2.モデル開発の論拠と工夫

### 2-1. 最終判断の根拠

| 運用条件 | V3.4 (ResNet50) | V12 (EfficientNetV2) | 最終判断 (0.40採用理由) |
| :--- | :--- | :--- | :--- |
| **Recall (検知率)** | 83% | **87.1%** | 検知率を向上させた。 |
| **Precision (誤報率)** | 54% | **92.4%** | 誤報率を劇的に改善。 AIアラートの信頼性を高め、再検査の工数削減（人員確保の課題に対応）を最優先した。 |
| **Loss (モデルの賢さ)** | 0.53 (限界) | **0.1683 (最高記録)** | Lossの最小化を監視することで、最も安定した「賢いモデル」を作成。 |

### 2-2. 学習手法の最適化

* **データ拡張:** ロゴの形状や向きはそのままに、回転 (0.2)、コントラスト/明るさ (0.05)のみを採用。
* **学習率スケジューラ:** **AdamW** を使用し、**Warm-up**と**Cosine Decay**を組み合わせたスケジュールを適用。学習の安定性と収束速度を改善した。
* **層の微調整:** `fine_tune.py` では `base_model` の後半 100層（約170層目以降）を解除し、低学習率で微調整を行い、ポテンシャルを引き出した。

***

## 3.環境構築と実行手順

### 3-1. ファイル構造（ローカルリポジトリ）

```text
2wins_practical_exam/
├── .gitignore                    # 仮想環境 (venv) と重いデータ (dataset/*) をGit管理から除外
├── train.py                      # V11 ベースモデル構築コード (EfficientNetV2)
├── fine_tune.py                  # V12 最終モデル構築コード (Loss監視)
├── evaluate.py                   # 最終評価・再現性の検証用コード (閾値0.40固定)
├── predict.py                    # 推論用コード (Threshold 0.40固定)
├── split_data.py                 # データ分割用コード
├── evaluation_matrix_final.png   # 最終成果の証拠画像（混同行列）
├── training_history_v12.png      # 最終学習グラフ
└── dataset/                      # (元データ。Git管理対象外)
```

### 3-2. モデルの再現手順



**前提:** モデル学習にはGPU環境 (Google Colabなど) が必要です。



1.  **データの分割** (`dataset.zip`を解凍後)

    ```bash

    python split_data.py

    ```

2.  **V11ベースモデルの学習** (土台作り)

    ```bash

    python train.py

    ```

3.  **V12最終モデルの微調整** (仕上げ)

    ```bash

    python fine_tune.py

    ```

4.  **最終評価の実行** (Recall 87.1%を証明)

    ```bash

    python evaluate.py

    ```

5.  **推論の実行** (新しい画像でテスト)

    ```bash

    python predict.py [画像ファイルのパス]

    ```

## 4. ☁️ 提出物とモデルファイルの配置
モデル本体は別途Google Driveに配置します。

* **最終モデル重みファイル名:** `v12_efficientnet_loss_best.keras`
* **Google Drive 共有リンク:** **(https://drive.google.com/file/d/1CYJefR5KLzoiGGG7WzVe9HG4YCSr9_f6/view?usp=sharing)**


***
