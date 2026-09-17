# 모델 구조

[소개](Home) · [실행 방법](Running)

## 1. 전체 구성

```text
RGB 사진 ────────────────┬─────────────────────────────┐
                        │                             │
               객체별 RGBA 마스크                Depth Anything 3
             (수동 편집 또는 SAM 3)             깊이·카메라·포인트맵
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                      객체마다 SAM 3D Objects 기반 생성
                                       │
                     Stage 1: Sparse Structure 생성
                       여러 시점의 정보를 가중 융합
                                       │
                     Stage 2: Structured Latent 생성
                       여러 시점의 정보를 가중 융합
                                       │
                            Mesh / Gaussian 디코딩
                                       │
                      DA3 좌표계 정렬·선택적 pose 최적화
                                       │
                               다중 객체 장면 병합
```

SAM 3는 **2D 마스크를 만드는 선택적 전처리 모델**, SAM 3D Objects는 **3D 생성 모델**입니다. 이름이 비슷하지만 서로 바꿔 쓸 수 없습니다.

## 2. 두 단계 생성

### Stage 1 — Sparse Structure

사진과 마스크 등의 조건을 바탕으로 객체가 차지하는 3D 공간의 희소 구조를 생성합니다. 다중 시점에서는 시점별 정보를 융합해 구조를 추정합니다.

### Stage 2 — Structured Latent (SLAT)

Stage 1의 구조를 바탕으로 형상·외관 정보를 담는 latent를 생성합니다. 이후 디코더가 mesh 또는 Gaussian 표현으로 변환합니다.

- Mesh: GLB로 저장, 일반적인 3D 편집기에서 열기
- Gaussian: PLY로 저장, Gaussian splat 지원 뷰어에서 확인

PLY를 일반 삼각형 mesh로 해석해서는 안 됩니다. 기본 `--decode_formats`는 `gaussian,mesh`이며, 다중 객체 GLB 병합에는 mesh 출력이 필요합니다.

## 3. 다중 시점 가중 융합

단순히 사진별 결과 mesh를 평균내는 방식이 아닙니다. 생성 과정에서 시점별 정보를 latent 단위로 가중 융합합니다.

1. **Warmup:** 단순 평균을 사용하는 첫 스텝으로 attention을 수집합니다.
2. **가중치 계산:** attention entropy 등을 바탕으로 시점별 가중치를 만듭니다. 낮은 entropy는 해당 구현에서 높은 신뢰도 신호로 사용되며, 실제 정확도를 보장하는 값은 아닙니다.
3. **본 생성:** 계산한 가중치를 사용해 처음부터 생성 과정을 수행합니다.

CLI 기본값:

| 항목 | 기본값 |
|---|---|
| Stage 1 weighting | 활성화 |
| Stage 1 entropy alpha | `30.0` |
| Stage 2 weighting | 활성화 |
| Stage 2 weight source | `entropy` |
| Stage 2 entropy alpha | `30.0` |

Stage 2는 `visibility`, `mixed` 방식도 지원합니다. 가시성 기반 처리는 깊이·카메라 정보를 필요로 합니다. 처음에는 기본 entropy 설정을 사용하세요. 한 시점만 입력하면 다중 시점 가중 융합 대신 단일 시점 경로를 사용합니다.

## 4. Multi-object는 어떻게 처리하는가?

`--mask_prompt red_cup,ball_bearings`를 주면 CLI가 쉼표로 구분된 폴더 이름을 읽고 다중 객체 모드로 진입합니다.

1. 같은 사진에서 `red_cup/` 마스크를 사용해 컵을 생성합니다.
2. 같은 사진에서 `ball_bearings/` 마스크를 사용해 구슬 그룹을 생성합니다.
3. 객체별 pose와 DA3 정보를 이용해 공통 좌표계로 배치합니다.
4. 요청한 경우 객체별 pose를 최적화합니다.
5. 객체 mesh를 하나의 장면으로 내보냅니다.

객체별 생성은 순차적이며, 물체 간 충돌이나 물리적 접촉을 보장하는 공동 생성·물리 시뮬레이션은 아닙니다. Pose 최적화는 DA3 포인트클라우드에 대한 정렬입니다. 기본적으로 회전·이동을 최적화하고, `--pose_opt_optimize_scale`을 켜야 스케일도 조정합니다.

## 5. 저 VRAM 모드

`--low_vram`은 모델 가중치를 CPU 메모리에 두고 필요한 단계에 GPU로 이동시켜 GPU 메모리 사용량을 줄입니다. 다른 모델로 교체하는 옵션은 아니며 CPU RAM과 데이터 이동 비용이 필요합니다. 이 모드에서는 모델 compilation을 비활성화합니다.

## 6. 코드 탐색 지도

링크는 Wiki 작성 시 확인한 커밋을 가리킵니다.

| 파일 | 역할 |
|---|---|
| [run_inference_weighted.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/run_inference_weighted.py) | CLI, 가중 추론, 객체 순차 처리, 결과 병합 |
| [scripts/run_da3.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/scripts/run_da3.py) | DA3 깊이·카메라 및 장면 출력 |
| [preprocessing/sam3_segmenter.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/preprocessing/sam3_segmenter.py) | 텍스트 기반 마스크 후보 선택 및 RGBA 저장 |
| [notebook/load_images_and_masks.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/notebook/load_images_and_masks.py) | 사진과 alpha 마스크 로딩 |
| [sam3d_objects/pipeline/inference_pipeline.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pipeline/inference_pipeline.py) | 2단계 생성·디코딩, 저 VRAM 관리 |
| [sam3d_objects/pipeline/multi_view_weighted.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pipeline/multi_view_weighted.py) | attention 수집 및 가중 융합 |
| [sam3d_objects/utils/latent_weighting.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/utils/latent_weighting.py) | entropy·visibility 가중치 계산 |
| [sam3d_objects/pose_align/pose_optimization.py](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/sam3d_objects/pose_align/pose_optimization.py) | 마스크 기반 포인트 추출 및 객체 정렬 |
