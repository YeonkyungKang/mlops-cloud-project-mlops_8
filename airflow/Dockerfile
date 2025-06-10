FROM python:3.11-slim-buster

ENV AIRFLOW_HOME=/usr/local/airflow

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y gcc libc-dev vim && \
    rm -rf /var/lib/apt/lists/*

# Install Apache Airflow and Docker Provider
#  - AIRFLOW_VERSION: 사용할 Airflow 버전 (2.10.0)
#  - PYTHON_VERSION: 사용할 Python 버전 (3.11)
#  - CONSTRAINT_URL: 안정적인 설치를 위한 제약 조건 파일 URL
ENV AIRFLOW_VERSION=2.10.0
ENV PYTHON_VERSION=3.11
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install Airflow and providers
RUN pip install "apache-airflow[celery,postgres,redis]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip install "apache-airflow-providers-docker" --constraint "${CONSTRAINT_URL}" --root-user-action=ignore

# 작업 디렉토리 생성
RUN mkdir -p $AIRFLOW_HOME/dags
WORKDIR $AIRFLOW_HOME

# DAG 및 엔트리포인트 복사
COPY /datapipeline_dag.py  $AIRFLOW_HOME/dags/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 포트 노출
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
