# 🤖 OpsCopilot — ログ異常検知システム

> サーバーログから異常を自動検知するMLシステムです。  
> 退勤後独学で開発中。

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-実験管理-orange)](https://mlflow.org/)
[![pytest](https://img.shields.io/badge/pytest-テスト済-green)](https://pytest.org/)

---

## 📌 概要

| 項目 | 内容 |
|------|------|
| 目的 | サーバーログの異常を自動検知し、運用チームの対応時間を短縮 |
| 手法 | Self-Attention（再構成誤差）+ IsolationForest |
| 現状 | プロトタイプ完成・実験管理・テスト導入済み |
| 進捗 | 6週目完了（Day 30 / 全195日） |

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

---

## 🏗️ アーキテクチャ
```
ログシーケンス (seq_len, d_model)
    → Self-Attention (Q=K=V=X)
    → 再構成誤差計算 (|X - output|.mean())
    → threshold 比較
    → 正常 / 異常 判定
```

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.14 |
| ML | scikit-learn, numpy |
| 実験管理 | MLflow |
| テスト | pytest |
| バージョン管理 | Git（ブランチ + PR方式） |

---

## 🚀 実行方法
```bash
# 依存関係インストール
pip install scikit-learn mlflow pytest numpy pandas

# 異常検知プロトタイプ実行
python notebooks/day28_logbert_proto.py

# threshold 探索
python notebooks/day29_threshold_search.py

# テスト実行
pytest notebooks/test_day29.py -v

# MLflow UI（結果確認）
mlflow ui
# → http://127.0.0.1:5000
```

---

## 📁 フォルダ構成
```
ops-copilot/
├── notebooks/
│   ├── day27_self_attention.py    # Self-Attention 実装
│   ├── day28_logbert_proto.py     # 異常検知プロトタイプ
│   ├── day29_threshold_search.py  # threshold 探索
│   └── test_day29.py              # pytest テスト
├── data/                          # ログデータ
└── README.md
```

---

## 💡 開発で意識していること
```
コードの可読性（可読性）  → 型ヒント・docstring・命名規則
再現性                    → random seed固定・MLflow実験記録
テスト習慣                → pytest・正常/境界値/エッジケース
Anti-Pattern回避          → マジックナンバー禁止・グローバル変数禁止
```

---

## 🗓️ 開発ロードマップ

- [x] Week 1-4: データパイプライン + IsolationForest baseline
- [x] Week 5  : MLflow 実験管理 + pytest 導入
- [x] Week 6  : Self-Attention 実装 + LogBERT プロトタイプ
- [ ] Week 7-8: LogBERT 本格学習 + FPR/Recall チューニング
- [ ] Week 9-12: RAG パイプライン構築
- [ ] Week 13-16: FastAPI + Docker サービス化
- [ ] Week 17-20: AWS デプロイ + CI/CD

---
