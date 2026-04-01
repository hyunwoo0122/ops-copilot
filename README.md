# 📚 ops-copilot — MLE 이직 준비 기록

> 일본 IT 기업 MLE 이직을 목표로 퇴근 후 1시간씩 공부한 기록입니다.

## 🎯 목표
- 목표 기업: 도쿄 IT 기업 MLE 포지션
- 현재 진행: 6주차 완료 (Day 30)

## 📁 폴더 구조
ops-copilot/
├── notebooks/   # 매일 Python 코드
├── data/        # 로그 데이터
└── docs/        # 학습 메모 + 회고

## 🛠️ 사용 기술
- Python, numpy, pandas, scikit-learn
- MLflow (실험 관리)
- pytest (테스트)
- Git (브랜치 + PR 방식)
- Self-Attention (numpy 구현)  

## 🚀 실행 방법
# 의존성 설치
pip install scikit-learn mlflow pytest numpy pandas

# 실험 실행
python notebooks/day23_mlflow_compare.py

# 테스트 실행
pytest notebooks/test_day24.py -v

# MLflow UI
mlflow ui

## 📊 5주차 주요 성과
| 날짜 | 내용 | 결과 |
|------|------|------|
| Day 22 | MLflow 첫 실험 | F1=0.65 |
| Day 23 | 실험 비교 (cont. 0.1 vs 0.2) | 최적=0.65 |
| Day 24 | pytest 테스트 3개 | 3 passed |

## 📊 6주차 주요 성과 (Day 26~30)
| 날짜   | 내용                        | 결과                          |
|--------|---------------------------|-------------------------------|
| Day 27 | Self-Attention numpy 구현  | Q·K.T/√d_k → softmax → ·V    |
| Day 28 | LogBERT 프로토타입          | 정상 0.2707 / 이상 0.4874 분리 |
| Day 29 | pytest 3개 + threshold 탐색 | 3 passed / 최적 threshold=0.1 |

## 🏗️ 현재 아키텍처 흐름
로그 시퀀스 (seq_len, d_model)
    → Self-Attention (Q=K=V=X)
    → 재구성 오류 계산 (|X - output|.mean())
    → threshold 비교
    → 정상 / 이상 판단

## 💡 1개월차 핵심 습관
1주차: 리팩터링    → 읽기 좋은 코드
2주차: 예외처리    → 안 죽는 코드
3주차: 하드코딩 금지 → 유연한 코드
4주차: 회귀 테스트  → 버그 재발 방지
