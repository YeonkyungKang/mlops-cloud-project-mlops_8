#!/bin/bash

# 최초 한 번만 metadata DB 초기화
airflow db init

# Admin 유저 생성 (이미 있으면 무시됨)
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Docker Connection 추가 (Airflow DockerOperator 사용을 위해 필수)
echo "[INFO] Setting Docker Connection 'docker_default'..."
airflow connections add 'docker_default' \
    --conn-type 'docker' \
    --conn-host 'unix://var/run/docker.sock' \
    || true # 이미 Connection이 존재하면 무시하고 계속 진행

# 백그라운드로 scheduler 실행
airflow scheduler &

# foreground로 webserver 실행 → PID 1을 차지해야 함
exec airflow webserver --port 8080
