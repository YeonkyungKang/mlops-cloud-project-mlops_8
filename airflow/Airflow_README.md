📘 Airflow_README.md

🐳 프로젝트 개요

이 프로젝트는 Apache Airflow를 Docker 환경에서 실행할 수 있도록 구성한 Dockerfile과 관련 스크립트(entrypoint.sh)를 포함합니다. 특히, Airflow Variable 백업(variables_backup.json) 및 declare -A 방식으로 정의된 변수들을 자동으로 Airflow 시스템에 등록하는 로직이 포함되어 있어, 설정된 변수들을 컨테이너 실행 시 자동으로 복원할 수 있습니다.

⸻

📁 포함 파일
	•	Dockerfile: Airflow 도커 이미지를 빌드하기 위한 설정 파일
	•	entrypoint.sh: 컨테이너 시작 시 실행되는 스크립트. Variable import 및 수동 설정 로직 포함
⸻

🛠️ Docker 이미지 빌드

docker build -t mlops-airflow .


⸻

🚀 컨테이너 실행

docker run -itd \
  --name mlops-airflow \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e AIRFLOW_UID=$(id -u) \
  mlops-airflow

옵션 설명:
	•	-itd: 백그라운드에서 인터랙티브하게 실행
	•	--name: 컨테이너 이름 지정
	•	-p 8080:8080: 웹 UI 포트 포워딩
	•	-v /var/run/docker.sock:/var/run/docker.sock: Docker-in-Docker 용 (선택)
	•	-e AIRFLOW_UID=$(id -u): 현재 사용자 UID 전달 (권한 오류 방지)

⸻

🔄 Variable 설정

Airflow UI 내에 Admin > Variables 아래 이름으로 변수 설정필요함.
S3 버킷 이름, Access 키, Regine 정보
DATA_PIPELINE_IMAGE : ML 파이프라인 도커 이름
