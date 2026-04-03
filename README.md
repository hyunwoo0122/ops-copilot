# 🤖 OpsCopilot — ログ異常検知システム

> サーバーログから異常を自動検知するMLシステムです。  
> 退勤後1時間、独学で開発中。

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-実験管理-orange)](https://mlflow.org/)
[![pytest](https://img.shields.io/badge/pytest-テスト済-green)](https://pytest.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-学習済-red)](https://pytorch.org/)

---

## 📌 概要

| 項目 | 内容 |
|------|------|
| 目的 | サーバーログの異常を自動検知し、運用チームの対応時間を短縮 |
| 手法 | PyTorch AutoEncoder + Self-Attention + IsolationForest |
| 現状 | AutoEncoder学習済・実験管理・テスト導入済み |
| 進捗 | 7週目完了（Day 35 / 全195日） |

---

## 🔍 解決する課題

- 手動ログ監視による見落としリスクを削減
- 障害対応時間の短縮
- 正常ログのみで学習し、ラベルなしで異常を検知（Self-Supervised）

---

## 📊 現在の成果（数字で証明）

### 5週目：実験管理基盤
| 内容 | 結果 |
|------|------|
| MLflow 初回実験 | F1 = 0.65 |
| contamination 比較（0.1 vs 0.2） | 最適 F1 = 0.65 |
| pytest テスト | 3 passed ✅ |

### 6週目：Attention ベース異常検知
| 内容 | 結果 |
|------|------|
| Self-Attention numpy 実装 | Q·Kᵀ/√d_k → softmax → ·V |
| LogBERT プロトタイプ | 正常スコア: 0.2707 / 異常スコア: 0.4874 ✅ |
| threshold 探索（0.1〜0.7） | MLflow で7実験記録 |
| pytest テスト | 3 passed ✅ |

### 7週目：PyTorch 学習ベース異常検知
| 内容 | 結果 |
|------|------|
| PyTorch AutoEncoder 実装 | 入出力 shape (batch, 8)→(batch, 8) 確認 |
| Early Stopping + モデル保存 | 過学習防止・best_model.pt |
| pytest テスト | 4 passed ✅ |

### 8週目：DataLoader + FPR/Recall チューニング
| 内容 | 結果 |
|------|------|
| DataLoader 導入（バッチ学習） | データ1000件・batch_size=32 |
| FPR/Recall/F1 測定 | threshold=0.7で最適化 |
| pytest テスト | 4 passed ✅（性能回帰テスト含む） |

---

## 🏆 全体の成果サマリー — 数字で証明

| フェーズ | 手法 | 正常誤差 | 異常誤差 | 倍率 | F1 |
|---------|------|---------|---------|------|-----|
| 6週目 | numpy（学習なし） | 0.2707 | 0.4874 | 1.8倍 | 0.33 |
| 7週目 | PyTorch AutoEncoder | 0.1646 | 6.9236 | 42倍 | - |
| 8週目 | DataLoader + チューニング | - | - | - | **0.995** |

→ threshold=0.7: Recall=0.990 / FPR=0.000 / F1=0.995

---

## 🏗️ 現在のアーキテクチャ
```
ログシーケンス (seq_len, d_model)
    → LogAutoEncoder (nn.Module)
        → Encoder: Linear(8→4) + ReLU
        → Decoder: Linear(4→8)
    → 再構成誤差計算 (MSELoss)
    → Early Stopping (patience=3)
    → threshold 比較
    → 正常 / 異常 判定
```

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.14 |
| ML | PyTorch, scikit-learn, numpy |
| 実験管理 | MLflow |
| テスト | pytest |
| バージョン管理 | Git（ブランチ + PR方式） |

---

## 🚀 実行方法
```bash
# 依存関係インストール
pip install torch scikit-learn mlflow pytest numpy pandas

# AutoEncoder 学習・評価
python notebooks/day33_logbert_train.py

# テスト実行
pytest notebooks/test_day34.py -v

# threshold 探索
python notebooks/day29_threshold_search.py

# MLflow UI（結果確認）
mlflow ui
# → http://127.0.0.1:5000
```

---

## 📁 フォルダ構成
```
ops-copilot/
├── notebooks/
│   ├── day27_self_attention.py     # Self-Attention 実装
│   ├── day28_logbert_proto.py      # 異常検知プロトタイプ
│   ├── day29_threshold_search.py   # threshold 探索
│   ├── day32_pytorch_autoencoder.py # PyTorch AutoEncoder
│   ├── day33_logbert_train.py      # Early Stopping + モデル保存
│   └── test_day34.py               # pytest テスト（4ケース）
├── data/                           # ログデータ
└── README.md
```

---

## 💡 開発で意識していること
```
コードの可読性  → 型ヒント・docstring・命名規則
再現性          → random seed固定・MLflow実験記録
テスト習慣      → pytest・正常/境界値/エッジケース
Anti-Pattern回避 → マジックナンバー禁止・グローバル変数禁止
過学習防止      → Early Stopping・モデル保存
```

---

## 🗓️ 開発ロードマップ

- [x] Week 1-4 : データパイプライン + IsolationForest baseline
- [x] Week 5   : MLflow 実験管理 + pytest 導入
- [x] Week 6   : Self-Attention 実装 + LogBERT プロトタイプ
- [x] Week 7   : PyTorch AutoEncoder + Early Stopping（42倍向上）
- [ ] Week 8   : LogBERT 本格学習 + FPR/Recall チューニング
- [ ] Week 9-12: RAG パイプライン構築
- [ ] Week 13-16: FastAPI + Docker サービス化
- [ ] Week 17-20: AWS デプロイ + CI/CD

---
