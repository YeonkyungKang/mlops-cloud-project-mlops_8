📌 ## 프로젝트 개요

<br>

## 💻 프로젝트 소개
### 서울시 관광지 기온 예측 서비스 (SeoulTourTempForecast)
- 서울시 주요 관광지의 기온을 예측하여 시민과 관광객에게 유용한 정보를 제공하는 기상 기반 관광 지원 서비스입니다. <br>
  기상청 API로 수집한 날씨 데이터를 전처리하고, 특정 관광지의 과거 기온 데이터를 기반으로 머신러닝 모델을 훈련시켜 예측 서비스를 제공합니다.
  이 프로젝트는 MLOps 관점에서 데이터 수집 → 모델 훈련 → 예측 서비스 제공까지의 전 과정을 자동화하는 것이 목적입니다.

### <작품 소개>
- _만드신 작품에 대해 간단한 소개를 작성해주세요_

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

<br>

## 🔨 개발 환경 및 기술 스택
- 주 언어 : Python 3.11
- 🧰 프레임워크 : Apache Airflow (workflow orchestration), FastAPI (API 서버), DVC (데이터 버전 관리)
- 버전 및 이슈관리 : _ex) github_
- 협업 툴 : _ex) github, notion_
- 📦 패키지 관리 : pip + requirements.txt
- 🧪 머신러닝 : scikit-learn, XGBoost
- 📁 데이터 버전 관리 : DVC
- 🧪 실험/모델 모니터링 : MLflow / Weights & Biases
- ☁️ 배포 환경 : AWS EC2 (Ubuntu), Docker
- 🔧 이슈관리 : GitHub Issues, GitHub Projects
- 🧑‍🤝‍🧑 협업 툴 : Notion, Slack, Google Drive
- 📄 환경 설정 : .env + dotenv

<br>

## 📁 프로젝트 구조
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

<br>

## 💻​ 구현 기능
### 🔄 데이터 수집
- 기상청 API를 활용한 기온/강수/습도 등 자동 수집
### 🧹 데이터 전처리
- 이상치 제거, 결측치 처리, Feature Engineering
### 📊 EDA
- 시각화를 통한 데이터 탐색 분석
### 🧠 모델링
- XGBoost, RandomForest 등으로 기온 예측 회귀 분석 모델 구축
### 📈 모델 성능 평가
- MAE, RMSE, R² 기준 성능 측정
### 💾 DVC 연동
- 데이터 및 모델 버전 관리
### ⚙️ 워크플로우 자동화
- Airflow DAG을 통한 주기적 수집/학습/예측 파이프라인
### 🚀 API 서비스
- FastAPI를 통해 특정 관광지의 예측 기온을 제공하는 API
### 📊 모니터링
- MLflow/W&B를 통한 실험 및 성능 추적
### 🐳 컨테이너화
- Dockerfile을 통한 이식성 높은 실행 환경 제공

<br>

## 🛠️ 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ub_u88a4MB5Uj-9Eb60VNA.jpeg)

<br>

## 🚨​ 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 📌 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 📰​ 참고자료
- _참고자료를 첨부해주세요_
