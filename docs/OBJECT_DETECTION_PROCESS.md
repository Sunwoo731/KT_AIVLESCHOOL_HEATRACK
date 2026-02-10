# 위성영상 객체판독 (Satellite Object Detection)

## 1. 개요 (Overview)
본 프로젝트는 YOLOv8-seg 모델을 사용하여 위성 영상(KOMPSAT-3/3A/5 등)에서 수계(Water), 건물, 도로 등을 식별하는 객체 탐지 및 분할(Instance Segmentation) 작업을 수행합니다.

## 2. 데이터셋 (Dataset)
학습 데이터는 원천 데이터(TIFF/PNG 이미지)와 라벨 데이터(마스크 이미지)로 구성됩니다.
YOLO 모델 학습을 위해 마스크 이미지는 Polygons 좌표(`x y x y ...`)가 포함된 `.txt` 형식으로 변환됩니다.

### 데이터 구조
- **원천 데이터**: `[원천]train_water_data/` (위성 이미지)
- **라벨 데이터**: `[라벨]train_water_labeling/` (Segmentation 마스크)

### 샘플 데이터
대용량 데이터는 GitHub 저장소에 포함되지 않았으며, 데이터 처리 과정을 이해할 수 있도록 샘플 데이터를 제공합니다.
- 경로: `data/sample/object_detection/`
  - `WTR00676_K5_NIA0209.tif`: 샘플 위성 영상
  - `WTR00676_K5_NIA0209_label.tif`: 해당 영상의 라벨(마스크)

## 3. 데이터 전처리 (Data Preprocessing)
`train_segmentation.py` 스크립트에서 다음과 같은 전처리가 수행됩니다:
1. **마스크 처리**: OpenCV를 사용하여 마스크 이미지(`_mask.png`, `_label.tif`)에서 윤곽선(Contours)을 추출합니다.
2. **좌표 변환**: 추출된 윤곽선 좌표를 이미지 크기(Width, Height)로 정규화(0~1 사이 값)합니다.
3. **YOLO 포맷 저장**: 클래스 ID와 정규화된 좌표를 `.txt` 파일로 저장합니다.
   - 포맷: `<class-id> <x1> <y1> <x2> <y2> ...`

## 4. 모델 학습 (Model Training)
- **모델**: YOLOv8n-seg (Nano Segmentation Model)
- **설정**:
  - Epochs: 1 (데모용 설정, 실제 학습 시 조정 필요)
  - Image Size: 512
  - Batch Size: 4
- **명령어**:
  ```python
  model = YOLO('yolov8n-seg.pt')
  model.train(data='data.yaml', epochs=1, imgsz=512, batch=4)
  ```

## 5. 참고 사항
- 원본 데이터셋은 용량이 매우 크므로(수 GB 이상), 로컬 환경에서 전체 데이터를 학습하려면 GPU가 권장됩니다.
- `.gitignore` 설정에 따라 원본 데이터 디렉토리는 업로드되지 않습니다.
