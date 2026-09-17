# MV-SAM3D 프로젝트 소개

MV-SAM3D는 여러 시점의 RGB 사진과 객체별 마스크로 3D 객체를 재구성하는 프레임워크입니다. SAM 3D Objects에 다중 시점 융합을 추가하고, Depth Anything 3(DA3)의 깊이·카메라 추정으로 객체를 장면에 배치합니다.

이 저장소는 [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D)의 포크입니다. 통합 설치 스크립트, `--low_vram` 옵션, 경로 및 실행 환경 관련 수정이 포함되어 있습니다.

## Wiki 안내

1. **프로젝트 소개** — 현재 페이지
2. [설치 및 환경 설정](Installation)
3. [데이터 준비](Data-Preparation)
4. [실행 방법: 이슈 #2로 multi-object 만들기](Running)
5. [모델 구조](Model-Architecture)

## 이번 예제: 빨간 컵 + 볼베어링

[이슈 #2](https://github.com/wyim-pgl/MV-SAM3D/issues/2)의 사진을 사용합니다.

<img src="https://github.com/user-attachments/assets/eb533ae9-0b53-4ee4-8d10-615c33bf209e" width="360" alt="이슈 2의 입력 사진: 빨간 컵과 볼베어링" />

- `red_cup`: 빨간 컵 하나
- `ball_bearings`: 줄지어 있는 볼베어링 전체를 하나의 그룹으로 취급
- 목표: 객체별 마스크 → DA3 → 객체별 3D 생성 → 두 객체를 한 장면으로 병합

**Multi-view**는 한 장면을 여러 시점에서 관찰하는 것이고, **multi-object**는 장면 안의 여러 객체를 각각 재구성하는 것입니다. 이 예제는 두 가지를 함께 사용합니다. 각 구슬을 독립 객체로 만들려면 구슬마다 별도의 마스크 폴더와 시점 간 일관된 ID가 필요합니다.

> 이슈 #2 사진으로 RTX 4090(24 GB)에서 실제 실행을 확인했습니다. SAM 1에 수동 지정한 박스로 마스크를 만든 뒤, DA3와 기본 multi-object 추론으로 **컵·구슬 그룹 두 mesh가 들어 있는 GLB**를 생성하고 렌더링했습니다. 기존 GPU pose 최적화의 메모리 부족 문제도 수정했으며, 수정본에서 **컵과 구슬 모두 최적화 성공**과 최적화된 GLB의 두 mesh를 확인했습니다. [실행 결과와 제한 사항](Running)을 참고하세요.

![실제 생성된 컵과 볼베어링 — 병합 장면 및 객체별 보기](assets/issue2-result-preview.jpg)

좌측은 두 객체를 병합한 장면, 가운데·우측은 각각의 canonical 모델입니다. 객체별 패널은 별도로 화면에 맞춰 확대했으므로 패널 간 크기를 비교하면 안 됩니다.

## 기능과 한계

- 단일 객체 및 다중 객체 재구성
- attention entropy 기반 다중 시점 가중 융합
- mesh(GLB), Gaussian splat(PLY) 출력
- DA3 장면 병합 및 선택적 pose 최적화
- 마스크 생성은 별도 전처리입니다. `--mask_prompt`는 분할 모델에 주는 문장이 아니라 **마스크 폴더명**입니다.
- 작은 금속 구슬, 반사 표면, 가림, 부족한 시점은 복원 품질을 떨어뜨릴 수 있습니다.
- 실제 치수의 자동 복원이나 계측 정확도를 보장하지 않습니다.

## 참고 자료

- [논문: MV-SAM3D](https://arxiv.org/abs/2603.11633)
- [프로젝트 README](https://github.com/wyim-pgl/MV-SAM3D/blob/main/README.md)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

기본 실행 기준: [`f608536`](https://github.com/wyim-pgl/MV-SAM3D/tree/f6085368ff52a8dbf7267e3acceef7fb19a31047). Pose 최적화 성공은 여기에 `pose_optimization.py`의 KNN 메모리 수정이 적용된 코드 기준입니다. 수정 코드·테스트·Wiki는 로컬 커밋으로 보관하며, 원격 GitHub에는 아직 push하지 않았습니다. [실제 작업 과정과 재현 방법](Running#7-이번-결과를-만든-실제-절차)에 사용한 명령과 검증 방법을 정리했습니다.
