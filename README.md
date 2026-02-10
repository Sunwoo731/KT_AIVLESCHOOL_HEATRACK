# HEATTRACK: 위성 기반 열수송관 이상 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**HEATTRACK**은 위성 영상을 활용하여 지하 열수송관의 이상 징후를 탐지하는 AI 시스템입니다. 아리랑 3A호(KOMPSAT-3A)와 Sentinel-2 위성 데이터를 융합하고, 머신러닝 기반의 이상 탐지 알고리즘을 적용하여 지하 매설 인프라의 잠재적 위험을 식별합니다.

## 🚀 핵심 기능

*   **위성 데이터 수집**: Google Earth Engine 및 CDSE API를 연동하여 관심 지역의 위성 영상을 자동으로 수집·전처리합니다.
*   **초해상화 (Downscaling)**: 저해상도 열영상을 고해상도로 변환하여 분석 정밀도를 높입니다.
*   **이상 탐지 (Anomaly Detection)**: AutoEncoder (MLPRegressor) 모델을 활용하여 정상 배관의 열적 패턴을 학습하고, 이를 벗어나는 이상 징후를 탐지합니다.
*   **파이프라인 자동화**: 데이터 수집부터 전처리, 모델 학습, 시각화까지의 과정을 통합 파이프라인으로 제공합니다.

---

## 🏗 프로젝트 구조

```bash
HEATTRACK/
├── configs/           # 설정 파일 (config.yaml)
├── data/              # 샘플 데이터 (object detection sample 등)
├── src/               # 소스 코드
│   ├── data/          # 데이터 수집 및 합성 데이터 생성
│   ├── models/        # AutoEncoder 모델 구현 (Scikit-Learn)
│   ├── processing/    # 영상 처리 및 초해상화 로직
│   ├── utils/         # 유틸리티 함수
│   ├── visualization/ # 결과 시각화 모듈
│   └── main.py        # 프로젝트 실행 엔트리포인트
├── docs/              # 문서 및 결과 이미지
│   └── OBJECT_DETECTION_PROCESS.md # 객체 탐지 프로세스 문서
├── GITHUB_UPLOAD_LIST.md # 업로드 된 파일 목록 및 설명
├── requirements.txt   # 의존성 패키지 목록
└── README.md          # 프로젝트 문서
```

## 📚 문서 및 데이터 (Documentation & Data)

본 저장소는 보안 및 용량 문제로 일부 원본 데이터를 제외하고 구성되었습니다. 각 파일에 대한 상세 설명은 [GITHUB_UPLOAD_LIST.md](GITHUB_UPLOAD_LIST.md)를 참고하세요.

### 위성영상 객체 탐지 (Object Detection)
아리랑 위성 영상을 활용한 수계 및 객체 탐지/분할 모델링 과정은 [docs/OBJECT_DETECTION_PROCESS.md](docs/OBJECT_DETECTION_PROCESS.md)에 상세히 기술되어 있습니다.
- **샘플 데이터**: `data/sample/object_detection/` 디렉토리에서 학습용 영상 및 라벨 샘플을 확인할 수 있습니다.

### 결과 시각화 (Visualization)
- 분석 결과 및 시각화 이미지는 `docs/images/` 디렉토리에 저장됩니다.

## 💻 설치 방법

1.  **레포지토리 클론**
    ```bash
    git clone https://github.com/Sunwoo731/HEATTRACK.git
    cd HEATTRACK
    ```

2.  **가상환경 생성 및 활성화 (권장)**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **의존성 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **설정 파일 확인**
    `configs/config.yaml` 파일에서 대상 지역, 날짜, 모델 파라미터 등을 설정할 수 있습니다.

---

## ⚡ 사용 방법 (Quick Start)

모든 기능은 `src.main` 모듈을 통해 실행됩니다.

### 1. 데이터 다운로드
설정된 지역(안양 등)의 위성 영상을 다운로드합니다.
```bash
python -m src.main download
```

### 2. 가상 파이프 데이터 생성 (시뮬레이션)
테스트를 위한 가상 배관망 데이터를 생성합니다.
```bash
python -m src.main simulate
```

### 3. 전체 파이프라인 실행
데이터 로드, 전처리, 이상 탐지 과정을 순차적으로 실행합니다.
```bash
python -m src.main pipeline
```

---

# 열수송관 누수 탐지 - 모델링 모듈

이 디렉토리는 위성 영상을 사용하여 열수송관 누수를 탐지하기 위한 핵심 머신러닝 및 데이터 처리 파이프라인을 포함하고 있습니다.

## 저장소 구성 (Repository Composition)
깃허브 저장소는 핵심 코드와 실행에 필요한 주요 데이터만으로 구성되어 있습니다.

### 1. 포함된 항목 (Included)
*   **핵심 소스 코드 (`src/`, `utils/`, `app/`)**: 데이터 수집, 전처리, 모델링, 결과 리포팅을 위한 전체 파이프라인 코드.
*   **유틸리티 스크립트 (`scripts/`)**: 실험 및 검증을 위해 정리된 보조 스크립트 모음.
*   **주요 데이터 (Key Data)**:
    *   `data/sample_arirang/`: 파이프라인 실행 테스트를 위한 아리랑 위성 샘플 데이터.
    *   `data/visuals/`: 모델링 결과 시각화 이미지 및 분석 리포트 예시.
*   **설명서 및 설정**: `README.md` (본 파일), `requirements.txt` (의존성 목록).

### 2. 제외된 항목 (Excluded)
저장소의 경량화와 효율적인 관리를 위해 다음 항목들은 `.gitignore`를 통해 자동 제외되었습니다.
*   **대용량 원천 데이터 (`data/raw/`)**: 수 기가바이트(GB) 단위의 위성 영상 파일.
*   **학습된 모델 가중치 (`*.pt`)**: 용량이 큰 모델 파일은 로컬에서 관리됩니다.
*   **가상환경 (`venv/`)**: 로컬 실행 환경 디렉토리.

## 디렉토리 구조
- **`src/`**: 핵심 애플리케이션 로직.
  - `data_collector.py`: 위성 데이터 수집.
  - `super_res.py`: SwinIR 기반 초해상도 모듈.
  - `anomaly_detect.py`: PatchCore 기반 이상 탐지.
  - `leak_seg.py`: YOLOv8 기반 누수 세분화.
- **`scripts/`**: 데이터 검증 및 유틸리티.
- **`models/`**: 모델 설정 파일 (가중치 제외).

## 설정 및 사용법
1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
2. 통합 파이프라인 실행:
   ```bash
   python main.py --target "YYYYMMDD 지역명"
   ```


## 📊 데이터 출처
*   **KOMPSAT-3A (아리랑 3A)**: 고해상도 열적외선 영상
*   **Sentinel-2**: 다중광광 영상 (Vegetation Index 등 활용)
*   **OpenStreetMap**: 도로망 및 건물 정보

## 🛡 라이선스
본 프로젝트는 MIT 라이선스를 따릅니다.
