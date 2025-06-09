FROM python:3.11.12-slim-bookworm

RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# 애플리케이션 코드를 위한 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 먼저 복사 (빌드 캐시 활용)
# requirements.txt 파일이 src/dataset 폴더에 직접 있다고 가정합니다.
# 빌드 컨텍스트가 'src'로 변경되었으므로, 'dataset' 폴더 내의 requirements.txt를 복사합니다.
COPY src/dataset/requirements.txt ./requirements.txt

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 'dataset' 폴더의 내용(Dockerfile, dataset 관련 .py 파일 등)을 이미지 내 /app 디렉토리로 복사합니다.
COPY src/dataset/ .

# 'util' 폴더의 내용을 이미지 내 /app/util 폴더로 복사합니다.
COPY src/util/ ./util/
