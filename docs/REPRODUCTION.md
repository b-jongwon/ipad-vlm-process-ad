# 재현·실행 안내

## 환경

실제로 사용한 환경은 Windows, Python3.12.10, RTX4060 8GB, torch2.7.1+cu128, torchvision0.22.1+cu128, transformers4.57.6, supervision0.27.0, numpy2.2.6입니다. feature encoder/검출기는 차례로 올려 GPU 메모리를 절약했습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r local_experiments/requirements.txt
.\.venv\Scripts\python.exe -m pip install supervision==0.27.0 threadpoolctl
```

CUDA 없는 환경에서 벤치마크용으로 가짜 feature를 생성하지 않습니다. 다른 GPU/라이브러리에서 부동소수점/학습 비결정성이 있어 동일 소수점 결과까지 보장하지 않습니다.

## 중요: 공개한 것과 로컬에만 남긴 것

- 공개: source, 정상 split/normal grammar, epoch/loss, calibration metadata, score 기반 지표, bootstrap, verification, chart, 빈 사람 리뷰 template.
- 비공개 로컬: IPAD ZIP, 실제 frame/crop, pretrained/학습 weights, numpy feature/score, raw API records와 응답 캐시, 비용 원장/키.
- `vlm_refinement`는 **캐시 기반 실험 runner**입니다. source clone만으로 기존 private runtime cache가 생기지 않습니다. 아래 prerequisite을 실제 데이터로 생성하거나 원래 실험 workspace에서 실행해야 합니다.
- API request를 새로 생성하면 응답과 phase가 달라질 수 있습니다. 같은 숫자의 정확한 replay는 같은 accepted response/cache/checkpoint가 있어야 합니다.

## 1. 데이터와 모델 준비

IPAD 제공 사이트에서 데이터 사용 조건을 확인하고 dataset ZIP을 받습니다. 현재 기본 경로는 `D:/종프 학습/IPAD_dataset.zip`이며 `config.py` 또는 관련 entrypoint의 `--zip` 옵션을 환경에 맞게 수정합니다. 이 저장소가 dataset을 대신 배포하지 않습니다.

```powershell
.\.venv\Scripts\python.exe local_experiments/ipad_data.py --zip "D:/종프 학습/IPAD_dataset.zip"
.\.venv\Scripts\python.exe local_experiments/extract_features.py --zip "D:/종프 학습/IPAD_dataset.zip" --scenes R01 R04 --encoders clip dino --stride 4 --max-train 5000
```

모델은 코드에 고정한 공식 Hugging Face repository/revision으로 받습니다. 초기 다운로드는 인터넷이 필요합니다. `full_pipeline.features`가 `HF_HUB_OFFLINE=1`을 기본 설정하므로 모델 다운로드를 마친 뒤 객체 추출을 합니다. 원래 frame feature 캐시는stride4이고, refinement는 그중ordinal%8==0인 관측을 선택합니다.

## 2. 실제 normal grammar / object / 직접 VLM 캐시

`results/R01/grammar.json`, `results/R04/grammar.json`은 이전 실제 정상 영상에서 생성한 grammar 기록입니다. 정상 이미지가 같지 않으면 해당 공장에 이 vocabulary가 맞는지 다시 확인해야 합니다.

원래 `mentor_pipeline.data.GRAMMARS`는 private run의 strong grammar 위치를 가리킵니다. 새 clone에서는 코드가 공개 `results/<scene>/grammar.json`으로 fallback합니다. 공개 grammar는 pretrained 모델 weight도, 사람 정답도 아닙니다.

`full_pipeline.features` 객체 추출에는 각 scene의 `local_experiments/cache/full_pipeline_v1/<scene>/grammar.json`이 필요합니다. 실제 object vocabulary로 검출/추적을 실행한 뒤 아래 캐시를 생성합니다.

```text
local_experiments/cache/features/<scene>_<clip|dino>_*/
  complete.json (max_train=5000), records.json, embedding.npy
  patches.npy (DINO only)
local_experiments/cache/full_pipeline_v1/<scene>/features/
  complete.json, records.json, objects.npy, boxes.npy,
  geometry.npy, ids.npy, classes.npy, confidence.npy
local_experiments/runs/mentor_direct_20261005/<scene>/
local_experiments/runs/mentor_direct_reference_20261005/<scene>/
  records.json, states.json, model/evaluation_scores.npz
```

기존 직접 VLM 두 버전의 데이터 선택은 normal FIT72/CAL32와 전체 eligible test stride32입니다. 이 캐시는 VLM이 실제 전체 frame/객체 crop을 본 accepted 응답으로 채워야 합니다. `semantics.py`를 통해 정형화하고 validate합니다. 임의 phase나 랜덤 embedding으로 대체하면 실제 평가가 아닙니다.

새 API 작업은 현재 프로젝트의 credential/누적 비용 정책을 확인한 뒤 `mentor_pipeline.run --help`에 명시된 명시적 live/API 옵션으로 실행합니다. **이 문서의 준비 과정을 실행했다고 유료 요청에 자동 동의한 것은 아닙니다.** 기존 비용 원장 없이 private 결과를 재사용했다고 주장하지 않으며, 새 요청의 비용 상한은 별도 설정합니다.

## 3. refinement 학습 및 객체 확장

모든 prerequisite이 실제 데이터로 준비되었을 때:

```powershell
.\.venv\Scripts\python.exe -m local_experiments.vlm_refinement.run
.\.venv\Scripts\python.exe -m local_experiments.vlm_refinement.objects
.\.venv\Scripts\python.exe -m local_experiments.vlm_refinement.verify
.\.venv\Scripts\python.exe -m local_experiments.vlm_refinement.report
```

`run`이24 head 학습, PCA/prototype/phase/process fitting과 평가를 수행합니다. existing checkpoint가 있으면 records SHA를 확인해 재사용합니다. 다른 실험 설정으로 덮어쓰기하지 말고 새 RUN/OUT 디렉터리를 지정합니다.

`objects`는 실제 bbox crop에서 DINO feature를 새로 추출하고 class별 memory/motion 모델을 fit합니다. 기존24 head를 다시 처음부터 학습하지 않습니다. 결과는 object extension score/metadata로 분리하고 aggregate table에 합칩니다.

새 실험 출력은 `output/vlm_refinement_20261006`과 `local_experiments/runs/vlm_refinement_20261006`에 생깁니다. 공개 snapshot은 `results/`입니다. `report`가 생성한 private review images를 Git에 올리지 않습니다.

## 4. 테스트와 의미

```powershell
.\.venv\Scripts\python.exe -m unittest local_experiments.vlm_refinement.tests local_experiments.mentor_pipeline.tests local_experiments.tests.test_protocol
```

27개 contract test는 누수/인과성/점수 계산/예산 등의 코드 동작을 검증합니다. synthetic test 결과는 IPAD 성능을 대신하지 않습니다. `verify`는 실제 학습 checkpoint와 실제 feature를 다시 사용해24 head의 score를 검증합니다.

이번에 제공한 숫자 결과는 동일한 외부 cache/state/weights를 사용한 로컬 실행 기록입니다. 완성형 설치 프로그램, 범용 deployment API, 새 factory adaptation 성능 검증까지 포함한 패키지는 아닙니다.
