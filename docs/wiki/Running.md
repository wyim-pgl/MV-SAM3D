# 실행 방법: 이슈 #2로 multi-object 만들기

[데이터 준비](Data-Preparation) · 다음: [모델 구조](Model-Architecture)

이 예제는 [이슈 #2](https://github.com/wyim-pgl/MV-SAM3D/issues/2)의 빨간 컵과 볼베어링을 대상으로 합니다. 먼저 [데이터 준비](Data-Preparation)의 사진 세 장과 **객체별 RGBA 마스크 여섯 장**을 준비하세요.

```text
사진 3장 + 컵 마스크 3장 + 구슬 그룹 마스크 3장
                       ↓
           DA3: 공통 깊이·카메라 추정
                       ↓
     객체별 3D 생성 + 선택적 pose 최적화
          (컵 → 구슬 그룹 순서로 처리)
                       ↓
             공통 좌표계에서 병합
```

> **실제 실행 확인:** RTX 4090(24 GB), PyTorch 2.5.1+cu121, 코드 `f608536`에서 이슈 #2 사진 3장과 [검증용 RGBA 마스크](Data-Preparation)를 사용했습니다. 기본 multi-object 실행은 약 2분 24초에 완료됐고 두 mesh를 포함한 GLB를 다시 로드·렌더링했습니다. SAM 3 자동 분할과 실측 정확도는 검증하지 않았습니다.

![실제 생성 결과: 좌측 병합 장면, 가운데 컵, 우측 구슬 그룹](assets/issue2-result-preview.jpg)

가운데와 우측은 개별 canonical 모델을 각각 확대해 보여 줍니다. 조명에 따라 원본과 색이 달라 보일 수 있으며, 구슬 개수·치수·장면 배치의 정확성을 보장하는 결과는 아닙니다.

## 1. 환경 및 입력 확인

`mvsam3d` 환경을 활성화하고 저장소 루트로 이동합니다. 별도 SAM 3 환경에서 마스크를 만들었다면 반드시 재구성 환경으로 돌아옵니다.

```bash
conda activate mvsam3d
# micromamba로 설치했다면: micromamba activate mvsam3d

test -f checkpoints/hf/pipeline.yaml
find data/issue2_cup_bearings -maxdepth 2 -type f -name '*.png' | sort
```

사진 세 장과 마스크 여섯 장이 나오는지 확인하고, 데이터 준비 페이지의 마스크 검사도 실행합니다.

## 2. DA3 실행 — 두 객체가 공유하는 장면

```bash
python scripts/run_da3.py \
  --image_dir ./data/issue2_cup_bearings/images \
  --output_dir ./da3_outputs/issue2_cup_bearings
```

핵심 출력은 `da3_outputs/issue2_cup_bearings/da3_output.npz`입니다. 같은 사진으로 계산한 깊이·카메라 정보를 컵과 구슬 그룹이 공유합니다. 다른 장면에서 만든 NPZ를 재사용하지 마세요.

```bash
test -f da3_outputs/issue2_cup_bearings/da3_output.npz
find da3_outputs/issue2_cup_bearings -type f -name '*.glb'
```

장면 병합에는 DA3가 내보낸 장면 GLB도 사용됩니다. 아래 예제는 기본 시각화 출력을 유지하며, `--no_vis`를 추가하지 않습니다.

실제 실행에서 사진의 종횡비 차이 때문에 DA3가 중앙 크롭 경고를 출력했고, 깊이 배열은 `(3, 504, 294)`였습니다. 따라서 장면 가장자리 정보가 일부 잘릴 수 있습니다. 입력·마스크를 임의로 따로 리사이즈하지 말고, 더 정확한 배치가 필요하면 같은 종횡비로 촬영한 일관된 장면을 사용하세요.

## 3. Multi-object 기본 실행

```bash
python run_inference_weighted.py \
  --input_path ./data/issue2_cup_bearings \
  --mask_prompt red_cup,ball_bearings \
  --da3_output ./da3_outputs/issue2_cup_bearings/da3_output.npz \
  --merge_da3_glb \
  --low_vram
```

| 인자 | 의미 |
|---|---|
| `--input_path` | `images/`, `red_cup/`, `ball_bearings/`를 포함하는 장면 경로 |
| `--mask_prompt red_cup,ball_bearings` | 쉼표로 구분한 두 마스크 폴더. 다중 객체 모드 선택 |
| `--da3_output` | 공통 장면의 깊이·카메라 정보 |
| `--merge_da3_glb` | 객체들을 DA3 장면과 함께 내보내기 |
| `--low_vram` | 모델 단계별 메모리 이동을 이용하는 저 VRAM 모드 |

코드는 객체를 순서대로 처리한 뒤 결과를 합칩니다. 이것은 사진 전체를 한 번에 분할하는 명령도, 여러 객체를 하나의 모델 호출로 동시에 생성하는 명령도 아닙니다.

**확인할 로그:** `red_cup`, `ball_bearings` 각각의 처리 완료와 `Merging 2 objects`. 현재 코드는 한 객체가 실패해도 다른 객체를 병합할 수 있으므로, 마지막 `COMPLETE` 메시지만으로 두 객체 모두 성공했다고 판단하면 안 됩니다.

## 4. Pose 최적화를 포함해 실행 — 메모리 수정 후 24 GB에서 검증

기존 코드(`f608536`)에서는 컵의 pose 최적화가 `CUDA out of memory`로 실패했습니다. 원인은 `torch.cdist`를 배치로 나누더라도 역전파를 위해 모든 배치의 거리 행렬이 남아 있었기 때문입니다.

수정본은 기존 의존성인 PyTorch3D `knn_points`로 최근접점 인덱스를 찾고, 선택된 점 사이의 Euclidean distance만 역전파합니다. 점 수·반복 횟수·손실의 정의를 줄이거나 바꾸지 않았으며, 초기 손실 확인에는 gradient를 보관하지 않습니다.

**수정본으로 RTX 4090에서 아래 명령을 그대로 재실행해 컵·구슬 모두 최적화 완료를 확인했습니다.** 전체 재구성·최적화·병합은 약 2분 35초였고, `result_multiobj_merged_optimized.glb`에 두 mesh가 포함되어 실제 렌더링도 통과했습니다. 10만 target × 5만 source 점의 독립 손실/역전파 테스트에서 추가 GPU 메모리는 18.4 MiB였습니다(전체 파이프라인 VRAM 사용량이 아님).

> 수정은 이 문서와 함께 로컬 커밋에 포함되며 검증 GPU 서버에도 적용했습니다. GitHub에는 아직 push하지 않았습니다. 원격의 기존 `f608536`만 받으면 수정이 포함되지 않습니다. 실패 시 종료 코드가 0이 되는 기존 동작은 이번 메모리 수정 범위에서 바꾸지 않았으므로, 다른 원인의 실패에도 객체별 로그와 최적화된 mesh 개수를 확인하세요.

![같은 실행에서 최적화 전후 비교 — 양쪽 모두 컵과 구슬 포함](assets/issue2-pose-fixed-preview.jpg)

기본 결과의 마스크·객체 형상을 확인한 다음, 메모리 수정이 적용된 코드에서 DA3 포인트클라우드와의 정렬을 실행합니다. 새 추론 결과가 별도 타임스탬프 폴더에 생성되며 이전 결과를 단순 후처리하는 명령은 아닙니다.

```bash
python run_inference_weighted.py \
  --input_path ./data/issue2_cup_bearings \
  --mask_prompt red_cup,ball_bearings \
  --da3_output ./da3_outputs/issue2_cup_bearings/da3_output.npz \
  --merge_da3_glb \
  --run_pose_optimization \
  --low_vram
```

기본 pose 최적화는 **회전·이동**을 조정합니다. 상대 스케일까지 최적화하려면 `--pose_opt_optimize_scale`을 추가합니다. 이는 DA3 장면에 대한 크기 정렬이지 mm 단위 보정이 아닙니다.

작은 구슬의 마스크는 기본 erosion(kernel=3)에서 줄어들 수 있습니다. 구슬의 유효 포인트가 거의 남지 않는 경우에만 마스크와 DA3 깊이를 먼저 확인하고 `--pose_opt_mask_erosion 1`을 비교해 보세요. 최적화가 부정확한 깊이 또는 마스크를 고쳐 주지는 않습니다.

## 5. 결과 열기

다중 객체 결과는 다음 위치에 생성됩니다. `<run>`은 옵션과 타임스탬프를 포함한 폴더명입니다.

```text
visualization/issue2_cup_bearings/multiobject/<run>/
├── red_cup/
├── ball_bearings/
├── result_multiobj_merged.glb
├── result_multiobj_merged_scene.glb
├── result_multiobj_merged_optimized.glb          # 최적화 결과가 있을 때
└── result_multiobj_merged_scene_optimized.glb    # 최적화 + DA3 병합 성공 시
```

| 파일 | 확인 목적 |
|---|---|
| `result_multiobj_merged.glb` | 컵과 구슬 그룹만 확인 |
| `result_multiobj_merged_scene.glb` | DA3 장면과 함께 위치 관계 확인 |
| `result_multiobj_merged_optimized.glb` | 최적화된 객체 배치 확인 |
| `result_multiobj_merged_scene_optimized.glb` | 최적화 결과를 DA3 장면과 함께 확인 |

파일은 해당 단계가 성공했을 때만 생성됩니다. 최적화 파일이 있어도 두 객체 모두 최적화됐다는 뜻은 아니므로 객체별 로그·출력도 확인합니다. 다중 객체 실행은 결과를 `<run>/<객체명>/`으로 복사한 뒤 원래 객체별 실행 폴더를 지울 수 있습니다. 전체 로그를 보존하려면 명령 끝에 `> run.log 2>&1`을 붙이세요.

이번 기본 실행의 `result_multiobj_merged.glb`는 **22,876,536 bytes**, mesh **2개**이며 컵 499,516 vertices, 구슬 그룹 72,368 vertices였습니다. 삼각형과 좌표가 존재하고 유한한 값인지 확인했으며, GLB를 실제 렌더링했습니다. 이는 파일 생성·표시 검증이지 계측 정확도 검증은 아닙니다.

```bash
find visualization/issue2_cup_bearings -type f -name '*.glb' | sort
```

Blender에서 **File → Import → glTF 2.0**으로 GLB를 엽니다. DA3 장면을 포함한 파일에는 원래 장면의 표면도 들어갈 수 있으므로, 객체가 중복되어 보이는지 확인할 때는 객체만 담긴 `result_multiobj_merged.glb`와 비교하세요.

확인 사항:

- 컵과 구슬 그룹이 모두 존재하는가?
- 테이블이나 배경이 객체에 붙지 않았는가?
- 구슬이 하나만 남거나 서로 합쳐지지 않았는가?
- 두 대상의 상대 위치·크기가 입력 사진과 크게 다르지 않은가?
- 최적화 결과가 원래 결과보다 실제로 나아졌는가?

## 6. 실제 크기 보정은 별도 작업

이슈 #2의 목표에는 컵과 볼베어링을 이용한 크기 보정도 포함되어 있지만, 이슈에는 실제 치수가 제공되지 않았습니다. 이 Wiki에서는 임의의 컵 높이·구슬 지름을 사용하지 않습니다.

실측 기준 길이를 알고 있다면 Blender 등에서 동일한 부위의 모델 길이를 측정하고 다음 비율로 **병합된 장면 전체**를 균일하게 확대·축소할 수 있습니다.

```text
스케일 계수 = 실측 기준 길이 / 모델에서 측정한 기준 길이
```

분자와 분모의 단위 표현을 맞추고, 전체 장면에 같은 계수를 적용해야 객체 간 상대 위치를 유지할 수 있습니다. 이는 후처리 보정이며 이 저장소가 실제 치수를 자동 복원한다는 의미가 아닙니다.

## 7. 이번 결과를 만든 실제 절차

### 7.1 원격 GPU와 준비 상태 확인

Lab의 SSH 설정에 등록된 `gpu` 별칭으로 접속했습니다. 별칭이 없는 환경에서는 먼저 Lab Wiki의 접속 절차를 따르세요.

```bash
ssh gpu
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv
```

실제 검증 환경은 RTX 4090 24 GB, PyTorch 2.5.1+cu121입니다. 서버에 이미 설치된 `mvsam3d` 환경과 SAM 3D Objects·DA3 체크포인트를 재사용했습니다. 다른 작업의 GPU 메모리를 확보하기 위해 프로세스를 종료하지는 않았습니다.

이후 명령은 **GPU 서버의 MV-SAM3D 저장소 루트, 활성화된 재구성 환경**에서 실행합니다. 신규 환경은 [설치 및 환경 설정](Installation)을 먼저 완료하세요.

### 7.2 마스크를 실제로 만든 방식

1. 이슈 #2의 처음 세 사진을 원본 크기로 받았습니다.
2. SAM 3가 설치되어 있지 않아, 캐시에 있던 `facebook/sam-vit-huge`(SAM 1)를 사용했습니다.
3. 사진마다 컵을 둘러싼 박스 하나, 각 구슬을 둘러싼 박스를 각각 수동으로 지정했습니다.
4. 각 박스의 마스크 후보 중 SAM 점수가 가장 높은 것을 선택했습니다.
5. 구슬별 마스크의 합집합을 `ball_bearings` 마스크로 저장했습니다.
6. 원본 RGB와 이진 alpha를 합친 RGBA PNG 여섯 장을 만들고, 초록색 오버레이로 눈으로 확인했습니다.

이것은 **SAM 1 기본 CLI의 자동 분할 결과도, SAM 3 텍스트 분할 결과도 아닙니다.** 사진별 좌표가 포함된 [실제 사용 스크립트](assets/prepare_issue2_masks.py)를 별도로 실행했습니다. 스크립트의 `local_files_only=True`는 모델이 이미 캐시에 있어야 한다는 뜻입니다.

가장 간단한 재현 방법은 [데이터 준비](Data-Preparation)에 있는 검증용 ZIP을 사용하는 것입니다. 직접 같은 마스크를 다시 만들려면 원본 사진을 준비한 뒤 저장소에 포함된 스크립트를 실행합니다.

```bash
python docs/wiki/assets/prepare_issue2_masks.py
```

ZIP에는 실제 사용한 사진·마스크뿐 아니라 박스 좌표와 마스크 면적을 기록한 `sam1_prompt_report.json`도 들어 있습니다.

### 7.3 메모리 오류를 어떻게 수정했는가?

기존 코드는 5,000개 target 점과 최대 50,000개 source 점의 거리를 한 배치에서 계산했습니다. 이 거리 행렬 하나만 float32 기준으로 약 **954 MiB**입니다. 배치를 나눠도 `loss.backward()`를 위해 이전 배치의 행렬이 남아, 컵의 pose 최적화에서 메모리 부족이 발생했습니다.

수정 파일: `sam3d_objects/pose_align/pose_optimization.py`.

```python
# 최근접점 선택은 미분할 필요가 없습니다.
with torch.no_grad():
    nearest = knn_points(
        self.target_points.unsqueeze(0),
        source_aligned.unsqueeze(0),
        K=1,
    ).idx[0, :, 0]

# 선택한 점과의 거리만 미분합니다. 제곱 거리가 아닌 원래 Euclidean 거리입니다.
cd_loss = torch.linalg.vector_norm(
    self.target_points - source_aligned[nearest], dim=1
).mean()
```

`from pytorch3d.ops import knn_points`를 추가했고, 학습률 결정에만 쓰는 초기 손실 계산도 `torch.no_grad()`로 감쌌습니다. 기존 PyTorch3D를 재사용했으므로 새 패키지는 추가하지 않았습니다. 점 수, 반복 횟수, scale 옵션, regularization은 그대로입니다.

이 문서와 함께 커밋된 코드에는 수정이 포함되어 있습니다. **과거 `f608536` 코드만 사용하는 경우에 한해**, [수정 패치](assets/pose-memory-fix.patch)를 저장소 루트에 내려받아 적용할 수 있습니다. 이미 수정된 코드에는 다시 적용하지 마세요.

```bash
# 과거 코드에만 적용하는 선택 단계
git apply --check pose-memory-fix.patch
git apply pose-memory-fix.patch
```

### 7.4 최종 검증에 사용한 실행 명령

마스크가 준비되면 깊이를 생성하고 pose 최적화를 포함해 실행합니다. 각 단계가 성공한 뒤 다음 단계로 넘어갑니다.

```bash
python scripts/run_da3.py \
  --image_dir ./data/issue2_cup_bearings/images \
  --output_dir ./da3_outputs/issue2_cup_bearings

python run_inference_weighted.py \
  --input_path ./data/issue2_cup_bearings \
  --mask_prompt red_cup,ball_bearings \
  --da3_output ./da3_outputs/issue2_cup_bearings/da3_output.npz \
  --merge_da3_glb \
  --run_pose_optimization \
  --low_vram > issue2-pose.log 2>&1
```

이번 실행은 GPU pose 최적화를 사용했습니다. `--pose_opt_device cpu`로 우회하거나 `--pose_opt_optimize_scale`을 켜지는 않았습니다.

### 7.5 성공 여부를 확인한 방법

종료 코드 0만 확인하지 않았습니다. 아래를 모두 확인했습니다.

- 컵과 구슬 각각 `[Pose Optimization] Complete!` 로그가 있음
- 두 객체의 `pose_optimization/optimized_params.npz`와 최적화된 GLB가 있음
- pose의 scale·rotation·translation이 모두 유한한 값임
- `result_multiobj_merged_optimized.glb`에 비어 있지 않은 mesh **2개**가 있음
- 각 mesh의 vertex 좌표가 유한하며 face가 있음
- EGL 기반 offscreen renderer로 최종 GLB를 실제 렌더링함

새 실행 직후 다음 검사로 핵심 결과를 확인할 수 있습니다. 같은 폴더에서 여러 실행을 병행했다면 `run`을 로그에 찍힌 정확한 경로로 바꿔 주세요.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import trimesh

root = Path('visualization/issue2_cup_bearings/multiobject')
run = sorted(p for p in root.iterdir() if p.is_dir())[-1]
for obj in ('red_cup', 'ball_bearings'):
    assert (run / obj / 'result_pose_optimized.glb').is_file(), obj
    with np.load(run / obj / 'pose_optimization/optimized_params.npz') as params:
        for key in ('scale', 'rotation', 'translation'):
            assert np.isfinite(params[key]).all(), (obj, key)
path = run / 'result_multiobj_merged_optimized.glb'
scene = trimesh.load(path, force='scene', process=False)
assert len(scene.geometry) == 2, 'Expected cup and bearings'
for mesh in scene.geometry.values():
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    assert np.isfinite(mesh.vertices).all()
print('Verified two optimized meshes:', path)
PY
```

추가로 회귀 테스트를 실행했습니다.

```bash
POSE_CUDA_STRESS=1 python -m pytest tests -q
```

결과는 **11 passed**였습니다. 새 테스트는 기존 dense 거리 계산과 손실값·pose gradient를 비교하고, 겹치는 점의 gradient가 유한한지, 역전파용 저장량이 선형 범위인지 검사합니다. 기존 코드에서는 메모리 회귀 테스트가 실패했고 수정 후 통과했습니다. CUDA 스트레스 테스트의 18.4 MiB는 **손실 계산·역전파의 추가 할당량**이며, 전체 추론의 최대 VRAM 수치가 아닙니다.

최종 최적화 GLB는 **22,877,004 bytes**, 두 mesh의 vertex 수는 **499,524 / 72,370**입니다. SHA-256:

```text
9f986ec4da76b710c0b870c886b02796415ce6823b2b7f8c11afde4680881c64
```

로컬 결과는 `artifacts/issue2-validation/result_multiobj_merged_optimized.glb`, 실제 전후 비교 그림은 위의 `issue2-pose-fixed-preview.jpg`입니다. 대용량 GLB와 전체 실행 로그는 로컬 검증 자료로 보관하며 Git 커밋에는 포함하지 않았습니다.

코드 근거: [다중 객체 실행·병합](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/run_inference_weighted.py#L2119-L2370), [pose 옵션](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/run_inference_weighted.py#L3986-L3998).
