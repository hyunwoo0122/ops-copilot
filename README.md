# 🤖 OpsCopilot — ログ異常検知システム

> サーバーログから異常を自動検知するMLシステムです。  
> 退勤後1時間、独学で開発中。

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-実験管理-orange)](https://mlflow.org/)
[![pytest](https://img.shields.io/badge/pytest-テスト済-green)](https://pytest.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-学習済-red)](https://pytorch.org/)
[![Gemini](https://img.shields.io/badge/Gemini-API連携-blue)](https://ai.google.dev/)

---

## 📌 概要

| 項目 | 内容 |
|------|------|
| 目的 | サーバーログの異常を自動検知し、ランブック検索で対応時間を短縮 |
| 手法 | PyTorch AutoEncoder + RAG（FAISS + BM25）+ LLM（Gemini） |
| 現状 | 異常検知 + RAGパイプライン + LLM連携 構築済み |
| 進捗 | 15週目完了（Day 75 / 全195日） |
---

## 🔍 解決する課題

- 手動ログ監視による見落としリスクを削減
- 障害対応時間の短縮
- 正常ログのみで学習し、ラベルなしで異常を検知（Self-Supervised）
- 日本語/韓国語の日常表現から技術用語へのマッピング（LLM活用）

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

### 9週目：RAGパイプライン構築
| 内容 | 結果 |
|------|------|
| sentence-transformers 導入 | 多言語モデルで韓国語対応 |
| FAISSベクトル検索 | IndexFlatIP + normalize_L2 |
| 文章単位chunking | 類似度 0.23 → 0.54 向上 ✅ |
| threshold フィルタリング | 無関係クエリを自動除外 ✅ |

### 10週目：RAG評価・モデル比較
| 内容 | 結果 |
|------|------|
| RAG評価セット構築 | 8問・正確度・根拠率測定 |
| ランブック拡張 | 3件→10件 |
| 多言語モデル | 正確度62.5% |
| 韓国語特化モデル(ko-sroberta) | 正確度75.0% ✅ |
| pytest | 9件通過 |

### 11週目：LLM連携 + テスト最適化
| 内容 | 結果 |
|------|------|
| Gemini API連携 | 無料枠内で動作 |
| キャッシング実装 | 重複API呼び出し削減 |
| 指数バックオフ | 429エラー自動リトライ |
| CrashLoopBackOff精度 | 0% → 100% ✅ |
| pytest Mock | API費用なしでテスト |

### 12週目：FastAPI + Docker サービス化
| 内容 | 結果 |
|------|------|
| FastAPI 基本構造 + ミドルウェア | リクエスト時間自動記録 ✅ |
| 非同期処理（asyncio） | 応答時間 14秒→4秒（70%短縮）✅ |
| Dockerコンテナ化 | ローカル1-click実行 ✅ |
| pytest TestClient + Mock | APIコストなしでテスト ✅ |

### 13週目：AWS デプロイ + CI/CD + IaC
| 内容 | 結果 |
|------|------|
| AWS IAM + ECR | Dockerイメージpush成功 ✅ |
| ECS Fargate デプロイ | http://18.181.226.89:8000/health 確認 ✅ |
| GitHub Actions CI/CD | mainブランチpush→ECR自動デプロイ ✅ |
| Terraform IaC | ECR + セキュリティグループをコードで管理 ✅ |

### 14週目：ロードテスト + キャッシング + モニタリング
| 内容 | 結果 |
|------|------|
| locust ロードテスト | 失敗率24%・p95 23秒 確認 ✅ |
| TTLキャッシング実装 | 同一クエリ2回目からキャッシュ返却 ✅ |
| CloudWatch メトリクス | ResponseTime送信 + アラーム設定 ✅ |
| pytest キャッシュ性能テスト | 3個全て通過 ✅ |

### 15週目：SQS非同期キュー + オートスケーリング + テスト
| 内容 | 結果 |
|------|------|
| SQS キュー生성 + メッセージ送受信 | ✅ |
| Producer/Consumer 役割分離 | ✅ |
| SQS Worker 無限ループ実装 | ✅ |
| ECS オートスケーリング Terraform | CPU70%・最小1・最大3 ✅ |
| SQS Mock pytest | 4個全て通過 ✅ |

---

## 🏆 全体の成果サマリー — 数字で証明

### 異常検知
| フェーズ | 手法 | 正常誤差 | 異常誤差 | 倍率 | F1 |
|---------|------|---------|---------|------|-----|
| 6週目 | numpy（学習なし） | 0.2707 | 0.4874 | 1.8倍 | 0.33 |
| 7週目 | PyTorch AutoEncoder | 0.1646 | 6.9236 | 42倍 | - |
| 8週目 | DataLoader + チューニング | - | - | - | **0.995** |

→ threshold=0.7: Recall=0.990 / FPR=0.000 / F1=0.995

### RAG精度改善
| フェーズ | 手法 | 正確度 |
|---------|------|-------|
| 10週目 | 多言語モデル | 62.5% |
| 10週目 | 韓国語特化モデル | 75.0% |
| 11週目 | LLM意図把握 + ハイブリッド検索 | CrashLoopBackOff 100% ✅ |

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

RAG + LLM 分析パイプライン
    ユーザークエリ（日常語）
        → LLM (Gemini API) - 意図把握
        → 日常語 → 技術用語に変換
        → ハイブリッド検索 (FAISS + BM25)
        → ベクトル検索 (意味的類似度)
        → BM25 (キーワードマッチング)
        → threshold フィルタリング
        → ランブック返却
```

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.14 |
| ML | PyTorch, scikit-learn, numpy |
| RAG | sentence-transformers, FAISS, BM25 |
| LLM | Gemini API (無料枠) |
| 実験管理 | MLflow |
| テスト | pytest |
| バージョン管理 | Git（ブランチ + PR方式） |
| API | FastAPI, uvicorn |
| インフラ | Docker, AWS ECR, ECS Fargate |
| CI/CD・IaC | GitHub Actions, Terraform |
| メッセージキュー | AWS SQS |

---

## 🚀 実行方法

```bash
# 依存関係インストール
pip install torch scikit-learn mlflow pytest numpy pandas
pip install sentence-transformers faiss-cpu rank-bm25
pip install google-generativeai python-dotenv

# .env ファイルを作成
echo "GEMINI_API_KEY=your_api_key" > .env

# 異常検知 学習・評価
python notebooks/day33_logbert_train.py

# RAG + LLM パイプライン実行
python notebooks/day53_llm_intent.py

# テスト実行
pytest notebooks/ -v

# MLflow UI（実験結果確認）
mlflow ui
# → http://127.0.0.1:5000
```

---

---

## 📁 フォルダ構成
```
ops-copilot/
├── notebooks/
│   ├── day27_self_attention.py       # Self-Attention 実装
│   ├── day28_logbert_proto.py        # 異常検知プロトタイプ
│   ├── day29_threshold_search.py     # threshold 探索
│   ├── day32_pytorch_autoencoder.py  # PyTorch AutoEncoder
│   ├── day33_logbert_train.py        # Early Stopping + モデル保存
│   ├── day41_rag_basic.py            # RAG基礎（cosine類似度）
│   ├── day42_rag_embed.py            # sentence-transformers + FAISS
│   ├── day43_rag_chunk.py            # chunking + threshold
│   ├── day44_rag_sentence_chunk.py   # 文章単位chunking
│   ├── day47_rag_expand.py           # ランブック10件 + 評価
│   ├── day51_hybrid_search.py        # FAISS + BM25ハイブリッド
│   ├── day52_generalized_eval.py     # シナリオ別汎用評価
│   ├── day53_llm_intent.py           # LLM意図把握 + RAG
│   ├── test_day34.py                 # pytest（異常検知）
│   ├── test_day48.py                 # pytest（RAG）
│   └── test_day54.py                 # pytest Mock（LLM）
│   ├── day56_fastapi_basic.py        # FastAPI 基本構造
│   ├── day57_fastapi_middleware.py   # ミドルウェア + エラー処理
│   ├── day58_fastapi_async.py        # 非同期処理 + Docker
│   ├── test_day59.py                 # pytest TestClient
├── .github/
│   └── workflows/
│       └── deploy.yml                # GitHub Actions CI/CD
├── terraform/
│   └── main.tf                       # Terraform IaC
├── Dockerfile                        # Dockerコンテナ設定
├── requirements.txt                  # パッケージ一覧
├── data/                             # ログデータ
├── .env                              # APIキー（Git管理外）
├── .gitignore                        # .env を除外
└── README.md
```

---

## 💡 開発で意識していること
```
コードの可読性    → 型ヒント・docstring・命名規則
再現性            → random seed固定・MLflow実験記録
テスト習慣        → pytest・Mock・正常/境界値/エッジケース
Anti-Pattern回避  → マジックナンバー禁止・グローバル変数禁止
過学習防止        → Early Stopping・モデル保存
API最適化         → キャッシング・指数バックオフ・폴백
一般化            → 特定ケースへのOverfitting防止
```

---

## 🗓️ 開発ロードマップ

- [x] Week 1-4 : データパイプライン + IsolationForest baseline
- [x] Week 5   : MLflow 実験管理 + pytest 導入
- [x] Week 6   : Self-Attention 実装 + LogBERT プロトタイプ
- [x] Week 7   : PyTorch AutoEncoder + Early Stopping（42倍向上）
- [x] Week 8   : LogBERT 本格学習 + FPR/Recall チューニング
- [x] Week 9   : RAG パイプライン構築（類似度0.54達成）
- [x] Week 10  : RAG評価・モデル比較（75%達成）
- [x] Week 11 : LLM連携・CrashLoopBackOff 0%→100%
- [x] Week 12 : FastAPI + Docker サービス化（応答時間70%短縮）
- [x] Week 13 : AWS ECR/ECS + GitHub Actions CI/CD + Terraform IaC
- [x] Week 14 : locust ロードテスト + TTLキャッシング + CloudWatch監視
- [x] Week 15 : SQS非同期キュー + ECSオートスケーリング + Mock pytest
- [ ] Week 16: 負荷テスト改善 + コスト最適化
- [ ] Week 17-20: AWS デプロイ最適化 + オートスケーリング

---
