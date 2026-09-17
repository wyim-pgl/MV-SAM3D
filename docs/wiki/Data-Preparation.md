# 데이터 준비

[설치](Installation) · 다음: [실행 방법](Running)

## 검증에 사용한 사진·마스크 받기

직접 마스크를 만들기 전에 **실제 GPU 검증에 사용한 입력**으로 재현하려면 [issue2-inputs.zip](assets/issue2-inputs.zip)을 내려받아 저장소 루트에 둡니다. 아래는 내려받은 ZIP 파일이 저장소 루트에 있다고 가정합니다.

```bash
sha256sum issue2-inputs.zip
# 371f9b9ad551d9e5f538f7bd2aa99af53e6dee18514391e6a18a8d360d438c69
unzip -n issue2-inputs.zip
```

ZIP에는 `data/issue2_cup_bearings/` 아래 원본 사진 3장, RGBA 마스크 6장, 프롬프트 기록 JSON이 들어 있습니다. 기존 파일은 덮어쓰지 않으므로 같은 이름의 다른 입력이 있다면 먼저 별도로 보관하세요. 압축 해제 후 아래 **세 시점 × 두 객체 검사**를 거쳐 [실행 방법](Running)으로 넘어갈 수 있습니다.

마스크는 SAM 3가 아닌 **SAM 1 + 사진별 수동 박스 지정**으로 생성했습니다. 각 구슬을 따로 분할한 후 합집합을 취했습니다. 사용한 [검증용 스크립트](assets/prepare_issue2_masks.py)는 모델 `facebook/sam-vit-huge`가 로컬 HF 캐시에 있을 때 실행 가능하며, 이 세 사진의 좌표에 한정됩니다. 일반 사진용 자동 다중 객체 분할기가 아닙니다.

![실제 추론에 사용한 여섯 마스크의 초록색 오버레이](assets/issue2-mask-overlays.jpg)

위 행은 컵, 아래 행은 구슬 그룹입니다. 원본을 확대해 보면 지정한 구슬 수가 시점별로 7/8/7개로 다릅니다. 가림·사진 간 변화 여부를 별도로 확인해야 하며, 같은 고정 장면의 계측 데이터라고 가정하지 마세요.

## 1. 이슈 #2에서 사용할 사진

출처: [Daily report 09/15/2026 part 2 — 이슈 #2](https://github.com/wyim-pgl/MV-SAM3D/issues/2).

첨부된 사진 중 **처음 세 장**을 사용합니다. 뒤에 나오는 유사 사진과 중복 이미지를 모두 별도 시점으로 넣지 않습니다.

| `images/0.png` | `images/1.png` | `images/2.png` |
|---|---|---|
| <img src="https://github.com/user-attachments/assets/3fd8d409-20cd-4fe0-86e1-7bb0a52644cd" width="220" alt="시점 0: 컵 내부와 볼베어링" /> | <img src="https://github.com/user-attachments/assets/eb533ae9-0b53-4ee4-8d10-615c33bf209e" width="220" alt="시점 1: 컵 측면과 볼베어링" /> | <img src="https://github.com/user-attachments/assets/1c969f4f-123e-48af-877c-64e0ed800641" width="220" alt="시점 2: 다른 방향에서 본 컵과 볼베어링" /> |

아래 명령은 GitHub 첨부 원본을 내려받습니다. 이 파일들은 **입력 RGB 사진이지 분할 마스크가 아닙니다**.

```bash
mkdir -p data/issue2_cup_bearings/images
curl -fL 'https://github.com/user-attachments/assets/3fd8d409-20cd-4fe0-86e1-7bb0a52644cd' \
  -o data/issue2_cup_bearings/images/0.png
curl -fL 'https://github.com/user-attachments/assets/eb533ae9-0b53-4ee4-8d10-615c33bf209e' \
  -o data/issue2_cup_bearings/images/1.png
curl -fL 'https://github.com/user-attachments/assets/1c969f4f-123e-48af-877c-64e0ed800641' \
  -o data/issue2_cup_bearings/images/2.png
```

다운로드 확인 시 해상도는 각각 **381×668, 502×668, 502×668**입니다. 각 사진과 그 사진의 마스크 해상도는 반드시 같아야 합니다. 사진을 변경·회전·크롭하면 마스크도 다시 만들고 DA3도 다시 실행하세요.

촬영 예제를 새로 만들 때에는 물체의 상대 위치를 고정하고 카메라만 이동하세요. 이슈 사진의 촬영 조건과 실제 치수는 검증되지 않았으며, 작은 반사 구슬과 제한된 시점 때문에 품질이 낮을 수 있습니다.

## 2. 완성해야 할 폴더 구조

```text
data/issue2_cup_bearings/
├── images/                 # 공통 입력 사진
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
├── red_cup/                # 컵만 남긴 RGBA 마스크
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
└── ball_bearings/          # 모든 구슬을 한 그룹으로 묶은 RGBA 마스크
    ├── 0.png
    ├── 1.png
    └── 2.png
```

이 예제의 `--mask_prompt red_cup,ball_bearings`는 위 두 폴더를 읽습니다. 텍스트만 입력한다고 마스크가 만들어지지는 않습니다.

## 3. 마스크 규칙

- 파일 형식: **RGBA PNG**
- RGB: 원본 사진의 색상 유지
- Alpha: 객체는 `255`, 배경은 `0`
- 파일명: 대응하는 사진과 같은 이름
- 해상도·위치: 원본과 동일한 전체 캔버스 유지, 객체만 잘라낸 작은 이미지 금지

로더는 `alpha > 0`을 전경으로 읽습니다. 일반 RGB 사진이나 흑백 마스크를 그대로 넣으면 전체 화면을 객체로 취급할 수 있습니다. 투명 배경으로 보인다고 가정하지 말고 실제 alpha 채널을 확인하세요.

### 방법 A: 직접 만든 RGBA 마스크

GIMP 같은 투명도 편집 도구에서 각 원본 사진을 엽니다.

1. alpha 채널을 추가합니다.
2. 컵의 보이는 영역을 선택하고 나머지를 투명하게 만듭니다. 컵 내부 표면과 테두리는 포함하되, 컵 밖 배경·그림자·구슬은 제외합니다.
3. 캔버스를 자르지 않고 `red_cup/0.png`로 내보냅니다.
4. 원본으로 돌아가 구슬 전체의 보이는 영역만 선택합니다. 구슬 사이의 배경까지 채우지 않습니다.
5. 나머지를 투명하게 만들고 `ball_bearings/0.png`로 내보냅니다.
6. `1.png`, `2.png`에도 반복합니다.

볼베어링 그룹의 마스크에는 분리된 여러 영역이 있어도 됩니다. 다만 이를 하나의 생성 대상처럼 복원하므로 구슬 개수나 개별 형상이 보존된다는 보장은 없습니다. 개별 구슬 복원이 목적이면 `bearing_01`, `bearing_02`처럼 나누고 모든 시점에서 같은 구슬에 같은 ID를 유지해야 합니다.

### 방법 B: SAM 3로 초안을 생성한 뒤 수정

[설치 페이지](Installation)의 SAM 3 준비가 끝난 환경에서, 저장소 루트 기준으로 실행합니다.

```bash
python preprocessing/build_mvsam3d_dataset.py \
  --input data/issue2_cup_bearings \
  --objects red_cup,ball_bearings \
  --sam3_root "$SAM3_ROOT" \
  --sam3_checkpoint "$SAM3_CHECKPOINT"
```

이 CLI는 객체 이름을 그대로 SAM 3 텍스트 프롬프트로 사용합니다. 결과가 부정확하면 수동 마스크로 수정하세요.

**볼베어링에서 특히 주의:** 현재 `SAM3MultiObjectSegmenter`는 각 사진에서 **점수가 가장 높은 마스크 하나만** 저장합니다. `ball_bearings`라고 입력해도 모든 구슬이 합쳐진다는 보장이 없습니다. 한 구슬만 선택되었다면 나머지 구슬을 직접 추가하거나 방법 A로 전체 그룹을 만드세요. SAM 1 기반 `sam_segmenter.py`는 텍스트 기반 다중 객체 선택 도구가 아니므로 이 예제의 두 마스크를 자동 생성하는 대체 명령으로 쓰지 않습니다.

전처리의 성공 메시지만으로 모든 시점이 준비됐다고 판단하지 마세요. 현재 SAM 3 래퍼는 일부 시점만 성공해도 객체 처리 성공으로 표시할 수 있습니다.

## 4. 세 시점 × 두 객체 검사

다음 코드는 파일 누락, RGBA 여부, 크기, 빈 마스크와 전체 화면 마스크를 확인합니다.

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

root = Path('data/issue2_cup_bearings')
for name in ('0', '1', '2'):
    with Image.open(root / 'images' / f'{name}.png') as image:
        size = image.size
    for obj in ('red_cup', 'ball_bearings'):
        path = root / obj / f'{name}.png'
        assert path.is_file(), f'Missing: {path}'
        with Image.open(path) as mask:
            assert mask.mode == 'RGBA', f'Not RGBA: {path}'
            assert mask.size == size, f'Size mismatch: {path}'
            alpha = np.asarray(mask)[..., 3]
        foreground = alpha > 0
        assert foreground.any(), f'Empty mask: {path}'
        assert not foreground.all(), f'Full-frame mask: {path}'
        print(f'{path}: foreground={foreground.mean():.2%}')
print('All six masks passed structural checks. Review them visually too.')
PY
```

마지막으로 여섯 마스크를 직접 열어 컵/구슬이 정확하게 분리됐는지 확인합니다. 위 검사는 잘못 선택한 객체나 누락된 구슬을 자동 판별하지 않습니다. 준비가 끝나면 [실행 방법](Running)으로 이동합니다.

코드 근거: [RGBA 로더](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/notebook/load_images_and_masks.py#L20-L44), [SAM 3 마스크 선택·저장](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/preprocessing/sam3_segmenter.py#L129-L189).
