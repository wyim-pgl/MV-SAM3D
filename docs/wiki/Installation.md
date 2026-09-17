# 설치 및 환경 설정

[소개](Home) · 다음: [데이터 준비](Data-Preparation)

## 1. 요구 환경

| 항목 | 요구 사항 |
|---|---|
| OS | Linux x86-64 |
| GPU | NVIDIA, CUDA 12.x 드라이버; sm_80 이상 권장 |
| GPU 메모리 | upstream 안내는 32 GB. 이 포크는 RTX 4090 24 GB에서 `--low_vram` 사용 기록이 있음 |
| 저장 공간 | 약 50 GB 여유 공간 |
| 환경 관리자 | micromamba, mamba 또는 conda |

24 GB 지원은 모든 입력 크기·시점 수에서의 성공을 보장하는 의미는 아닙니다. 시스템 CUDA toolkit은 필수가 아니며 설치 환경에서 CUDA 12.1 toolkit을 준비합니다.

## 2. 체크포인트 접근 권한

서로 다른 모델을 구분하세요.

| 모델 | 용도 | 준비 |
|---|---|---|
| SAM 3D Objects | 3D 재구성 | [Hugging Face 접근 승인](https://huggingface.co/facebook/sam-3d-objects) 필요 |
| Depth Anything 3 | 깊이·카메라 추정 | 첫 실행에서 자동 다운로드 |
| SAM 3 | 선택적 텍스트 기반 마스크 생성 | 별도의 [접근 승인](https://huggingface.co/facebook/sam3), 설치 및 체크포인트 필요 |

SAM 3D Objects 승인만으로 SAM 3가 준비되지는 않습니다. 직접 만든 RGBA 마스크가 있으면 SAM 3는 없어도 됩니다.

## 3. 설치

처음 설치할 때는 **환경 설치 → HF 인증 → 체크포인트 다운로드** 순서로 진행합니다.

```bash
git clone https://github.com/wyim-pgl/MV-SAM3D.git
cd MV-SAM3D
./install.sh --skip-checkpoints
```

설치 스크립트는 기본적으로 `mvsam3d` 환경을 만들고 SAM 3D Objects와 DA3를 설치합니다. 여기서는 인증 전에 다운로드가 시도되지 않도록 체크포인트를 건너뜁니다. 옵션 없이 `./install.sh`를 실행했다가 인증 오류로 종료된 경우에도 아래 활성화·인증·다운로드 단계를 진행할 수 있습니다.

설치에 사용한 환경 관리자로 활성화합니다. 다음 중 **하나만** 실행하세요.

```bash
conda activate mvsam3d
# 또는: micromamba activate mvsam3d
# 또는: mamba activate mvsam3d
```

이후 명령은 모두 **활성화된 환경의 저장소 루트**에서 실행합니다.

접근 승인 후 체크포인트를 준비합니다. 기존 `checkpoints/hf`가 있는 경우 중복 실행하지 마세요.

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download --repo-type model --local-dir checkpoints/hf-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
```

확인:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
test -f checkpoints/hf/pipeline.yaml && echo 'SAM 3D configuration found'
```

`CUDA available: False`이거나 설정 파일이 없으면 추론 전에 환경과 체크포인트를 먼저 수정합니다.

## 4. SAM 3 마스크 전처리 선택 시

[SAM 3 공식 설치 안내](https://github.com/facebookresearch/sam3)를 따라 SAM 3가 실행 가능한 환경과 `sam3.pt`를 준비합니다. `./install.sh`가 SAM 3까지 설치해 주는 것은 아닙니다. SAM 3의 요구 의존성이 재구성 환경과 다르면 **별도 전처리 환경**에서 마스크를 생성하고, 추론 시 `mvsam3d` 환경으로 돌아오세요.

전처리 실행 환경에서 실제 경로로 설정합니다.

```bash
export SAM3_ROOT=/absolute/path/to/sam3
export SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
python -c "import sys, os; sys.path.insert(0, os.environ['SAM3_ROOT']); import sam3; print('SAM 3 import OK')"
test -f "$SAM3_CHECKPOINT"
```

`SAM3_ROOT`는 소스 탐색 경로일 뿐, SAM 3 의존성을 자동으로 설치하지는 않습니다. 수동 RGBA 마스크를 사용할 경우 이 단계 전체를 건너뜁니다.

## 5. 먼저 기본 예제 확인

```bash
./examples/quickstart.sh
```

저장소에 포함된 `data/example`은 **단일 객체** 예제입니다. 먼저 설치가 정상인지 확인하고, [이슈 #2 다중 객체 예제](Running)로 넘어가세요.

상세 수동 설치 및 오류별 안내: [INSTALL.md](https://github.com/wyim-pgl/MV-SAM3D/blob/f6085368ff52a8dbf7267e3acceef7fb19a31047/INSTALL.md).
